import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    import nccl_ops
except ImportError:
    nccl_ops = None


class Zero3LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_shard, weight_shape, dp_group, dp_size, native_enabled):
        if dp_size == 1:
            weight = weight_shard.view(weight_shape)
        else:
            shards = [torch.empty_like(weight_shard) for _ in range(dp_size)]
            dist.all_gather(shards, weight_shard.contiguous(), group=dp_group)
            weight = torch.cat(shards).view(weight_shape)
        ctx.save_for_backward(x, weight_shard)
        ctx.weight_shape = weight_shape
        ctx.dp_group = dp_group
        ctx.dp_size = dp_size
        ctx.native_enabled = native_enabled
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_shard = ctx.saved_tensors
        if ctx.dp_size == 1:
            weight = weight_shard.view(ctx.weight_shape)
        else:
            shards = [torch.empty_like(weight_shard) for _ in range(ctx.dp_size)]
            dist.all_gather(shards, weight_shard.contiguous(), group=ctx.dp_group)
            weight = torch.cat(shards).view(ctx.weight_shape)
        grad_weight = grad_output.transpose(0, 1).matmul(x).contiguous()
        if ctx.dp_size == 1:
            grad_weight_shard = grad_weight.reshape_as(weight_shard)
            grad_input = grad_output.matmul(weight)
        else:
            grad_weight_shard = torch.empty_like(weight_shard)
            if ctx.native_enabled:
                stream_ptr = nccl_ops.reduce_scatter_async(grad_weight, grad_weight_shard)
                grad_input = grad_output.matmul(weight)
                comm_stream = torch.cuda.ExternalStream(stream_ptr, device=x.device)
                torch.cuda.current_stream(x.device).wait_stream(comm_stream)
            else:
                work = dist.reduce_scatter_tensor(
                    grad_weight_shard,
                    grad_weight.reshape(-1),
                    op=dist.ReduceOp.SUM,
                    group=ctx.dp_group,
                    async_op=True,
                )
                grad_input = grad_output.matmul(weight)
                work.wait()
            grad_weight_shard.div_(ctx.dp_size)
        return grad_input, grad_weight_shard, None, None, None, None


class Zero3Linear(nn.Module):
    def __init__(self, local_weight, dp_group, dp_size, native_enabled):
        super().__init__()
        if local_weight.numel() % dp_size:
            raise ValueError("Weight size must be divisible by the data-parallel size")
        self.weight_shape = tuple(local_weight.shape)
        self.weight = nn.Parameter(local_weight.reshape(-1).chunk(dp_size)[dist.get_rank(dp_group)].contiguous())
        self.dp_group = dp_group
        self.dp_size = dp_size
        self.native_enabled = native_enabled

    def forward(self, x):
        return Zero3LinearFunction.apply(
            x, self.weight, self.weight_shape, self.dp_group, self.dp_size, self.native_enabled
        )


class AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        output = x.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def rank_at(data_rank, pipe_rank, tensor_rank, pp_size, tp_size):
    return data_rank * pp_size * tp_size + pipe_rank * tp_size + tensor_rank


def create_mesh(dp_size, pp_size, tp_size):
    rank = dist.get_rank()
    data_rank = rank // (pp_size * tp_size)
    pipe_rank = (rank % (pp_size * tp_size)) // tp_size
    tensor_rank = rank % tp_size
    dp_group = pp_group = tp_group = None
    dp_ranks = pp_ranks = tp_ranks = None

    for p in range(pp_size):
        for t in range(tp_size):
            ranks = [rank_at(d, p, t, pp_size, tp_size) for d in range(dp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                dp_group, dp_ranks = group, ranks

    for d in range(dp_size):
        for t in range(tp_size):
            ranks = [rank_at(d, p, t, pp_size, tp_size) for p in range(pp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                pp_group, pp_ranks = group, ranks

    for d in range(dp_size):
        for p in range(pp_size):
            ranks = [rank_at(d, p, t, pp_size, tp_size) for t in range(tp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                tp_group, tp_ranks = group, ranks

    return data_rank, pipe_rank, tensor_rank, dp_group, pp_group, tp_group, dp_ranks, pp_ranks, tp_ranks


def make_weight(out_features, in_features, tensor_rank, tp_size, partition, seed, dp_size, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    weight = torch.randn(
        out_features, in_features, generator=generator, device=device
    ) / math.sqrt(in_features)
    if partition == "output":
        weight = weight.chunk(tp_size, dim=0)[tensor_rank]
    else:
        weight = weight.chunk(tp_size, dim=1)[tensor_rank]
    if weight.numel() % dp_size:
        raise ValueError("Tensor-parallel weight size must be divisible by the data-parallel size")
    return weight.contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--output-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.pp != 2:
        raise ValueError("This minimal pipeline schedule requires --pp 2")
    if args.hidden_size % args.tp:
        raise ValueError("hidden-size must be divisible by tp")
    if args.input_size % args.tp or args.output_size % args.tp:
        raise ValueError("input-size and output-size must be divisible by tp")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        world_size = dist.get_world_size()
        if world_size != args.dp * args.pp * args.tp:
            raise ValueError("World size must equal dp * pp * tp")

        (
            data_rank,
            pipe_rank,
            tensor_rank,
            dp_group,
            pp_group,
            tp_group,
            dp_ranks,
            pp_ranks,
            tp_ranks,
        ) = create_mesh(args.dp, args.pp, args.tp)

        if pipe_rank == 0:
            local_weight = make_weight(
                args.hidden_size,
                args.input_size,
                tensor_rank,
                args.tp,
                "output",
                args.seed,
                args.dp,
                device,
            )
        else:
            local_weight = make_weight(
                args.output_size,
                args.hidden_size,
                tensor_rank,
                args.tp,
                "input",
                args.seed + 1,
                args.dp,
                device,
            )

        native_enabled = nccl_ops is not None and args.dp > 1
        if native_enabled:
            dp_rank = dist.get_rank(dp_group)
            unique_ids = [nccl_ops.get_unique_id() if dp_rank == 0 else None]
            dist.broadcast_object_list(unique_ids, src=dp_ranks[0], group=dp_group)
            nccl_ops.init(unique_ids[0], dp_rank, args.dp, device.index)

        layer = Zero3Linear(local_weight, dp_group, args.dp, native_enabled).to(device)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=args.lr)
        peer = rank_at(data_rank, 1 - pipe_rank, tensor_rank, args.pp, args.tp)

        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator(device=device).manual_seed(
                args.seed + 100003 * data_rank + step
            )
            x = torch.randn(
                args.batch_size, args.input_size, generator=generator, device=device
            )
            target = torch.randn(
                args.batch_size, args.output_size, generator=generator, device=device
            )

            if pipe_rank == 0:
                activation = F.relu(layer(x))
                dist.send(activation.detach().contiguous(), dst=peer)
                grad = torch.empty_like(activation)
                dist.recv(grad, src=peer)
                activation.backward(grad)
            else:
                activation = torch.empty(
                    args.batch_size,
                    args.hidden_size // args.tp,
                    device=device,
                )
                dist.recv(activation, src=peer)
                activation.requires_grad_()
                partial_output = layer(activation)
                output = AllReduceSum.apply(partial_output, tp_group)
                loss = F.mse_loss(output, target)
                loss.backward()
                dist.send(activation.grad.contiguous(), dst=peer)
                if data_rank == 0 and tensor_rank == 0:
                    print(f"step={step} loss={loss.item():.6f}", flush=True)

            optimizer.step()
    finally:
        if nccl_ops is not None:
            nccl_ops.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
