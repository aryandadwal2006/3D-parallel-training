# Megatron-Style 3D Parallel Training

A minimal PyTorch training framework using a three-dimensional process mesh.

## Parallelism
- Rank layout: data, pipeline, tensor.
- Tensor parallelism shards the hidden dimension across tensor ranks.
- Pipeline parallelism splits the model into two stages with activation and gradient transfers.
- ZeRO-3 shards parameters and optimizer state across data ranks.
- NCCL all-gather reconstructs parameters; asynchronous reduce-scatter averages gradients.
- Gradient computation overlaps the reduce-scatter operation.

## Run
`DP=1 TP=1 ./launch.sh --steps 10`
