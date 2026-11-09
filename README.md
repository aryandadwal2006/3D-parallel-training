# Megatron-Style 3D Parallel Training

A PyTorch 3D-mesh trainer with tensor, pipeline, and ZeRO-3 data parallelism.
## Parallelism
- Rank layout: data, pipeline, tensor; pipeline uses two stages.
- Tensor parallelism shards hidden features; stages exchange activations and gradients.
- ZeRO-3 shards parameters and optimizer state across data ranks.
- NCCL all-gather restores weights; asynchronous reduce-scatter overlaps with input-gradient computation.
- Optional C++/pybind11 NCCL binding with a PyTorch distributed fallback.
## Launch
Local: `DP=1 TP=1 bash launch.sh --steps 10`; native build: `python setup.py build_ext --inplace`.
MPI: `MASTER_ADDR=$(hostname -s) mpirun -np 4 --map-by ppr:2:node bash launch_mpi.sh --dp 2 --pp 2 --tp 1`.
Slurm: `GPUS_PER_NODE=2 srun -N 2 --ntasks-per-node=1 bash launch_slurm.sh --dp 2 --pp 2 --tp 1`.
