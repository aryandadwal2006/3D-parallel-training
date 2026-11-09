# Megatron-Style 3D Parallel Training

A minimal PyTorch framework using a three-dimensional process mesh.
## Parallelism
- Rank layout: data, pipeline, tensor.
- Column/row tensor-parallel layers shard the hidden dimension.
- Two pipeline stages exchange activations and gradients.
- ZeRO-3 shards parameters and optimizer state across data ranks.
- NCCL all-gather rebuilds weights; asynchronous reduce-scatter averages gradients while input-gradient computation overlaps.
- Optional C++/pybind11 NCCL binding with a PyTorch distributed fallback.

## Run
`DP=1 TP=1 ./launch.sh --steps 10`
`python setup.py build_ext --inplace` builds the native operation.
` srun --ntasks-per-node=1 ./launch_slurm.sh --dp 2 --pp 2 --tp 1` runs under Slurm.
