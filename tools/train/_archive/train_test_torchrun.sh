#!/bin/bash
#SBATCH --job-name=TorchrunExample   # Job name
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --gpus-per-node=4            # GPUs per node
#SBATCH --ntasks-per-node=1          # 1 srun task per node
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH -p grete                     # Partition/queue
#SBATCH -A nib00034
#SBATCH -C inet
#SBATCH --exclude=ggpu114            # Example: exclude node ggpu114
#SBATCH -t 0-48:00:00
#SBATCH -o /user/henrich1/u12041/output/job-%J.out

#### (Optional) Proxy settings
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"

#### Performance hints (optional)
# export NCCL_NSOCKS_PERTHREAD=4
# export NCCL_SOCKET_NTHREADS=2
# export NCCL_MIN_CHANNELS=32

#### Rendezvous environment
export RDZV_HOST=$(hostname)     # use this node's hostname as rendezvous
export RDZV_PORT=29400           # pick a free port
echo "Running on host: $RDZV_HOST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"

#### Activate conda
echo "Activating conda..."
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate motip
cd ~/repos/MOTIP

#### Launch via srun + torchrun
# srun will start 1 task per node (due to --ntasks-per-node=1),
# and each task executes the same command below.
# torchrun then spawns --nproc_per_node=4 processes on each node, 
# for a total of 8 processes across 2 nodes.
srun torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node=4 \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    main.py \
      --mode train \
      --config ./configs/r50_ddetr_dt_train.yaml \
      --exp-name r50_ddetr_dt_train_multi_node_srun \
      --use-wandb False \
      --outputs-per-step 10


# AT LEAST THIS SCRIPT WORKED. But performance was bad