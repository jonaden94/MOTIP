#!/bin/bash
##SBATCH -G A100:1
#SBATCH -p grete
#SBATCH --nodes=2                # node count
#SBATCH --gpus-per-node=A100:4   # total number of gpus per node
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -A nib00034
#SBATCH -C inet
#SBATCH -o /user/henrich1/u12041/output/job-%J.out
#SBATCH --mem=256G
#SBATCH -t 0-48:00:00
##SBATCH --exclude=ggpu150,ggpu151,ggpu155,ggpu156
##SBATCH --mail-type=begin            # send mail when job begins
##SBATCH --mail-type=end              # send mail when job ends
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export HTTP_PROXY="http://www-cache.gwdg.de:3128"

echo "Activating conda..."
CONDA_BASE=$(conda info --base)
echo $CONDA_BASE
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate motip
cd ~/repos/MOTIP

####################################### srun
# These environment variables are required for initializing distributed training in pytorch  
export MASTER_PORT=29400
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$SLURM_NTASKS
echo "MASTER_ADDR="$MASTER_ADDR
# NCCL_IB_DISABLE=1
# CUDA_LAUNCH_BLOCKING=1
# TORCH_DISTRIBUTED_DEBUG=DETAIL 
# NCCL_DEBUG=INFO 

export WANDB_DISABLE_GIT=true
################# debug
# srun python main.py --mode train --config ./configs/r50_ddetr_dt_train.yaml --exp-name debug --use-wandb False --outputs-per-step 10 --pretrain /user/henrich1/u12041/repos/MOTIP/datasets/pretrained/full_model/r50_deformable_detr_motip_dancetrack_trainval_joint_ch.pth
# --use-wandb True --outputs-per-step 10
################# regular training
# srun python main.py --mode train --config ./configs/r50_ddetr_dt_trainval_joint_ch.yaml --exp-name r50_ddetr_dt_trainval_joint_ch_try
# srun python main.py --mode train --config ./configs/r50_ddetr_dt_train.yaml --exp-name r50_ddetr_dt_train_with_data_on_grete
srun python main.py --mode train --config ./configs/r50_ddetr_dt_train.yaml --exp-name r50_ddetr_dt_train_test_eval_time
# srun python main.py --mode train --config ./configs/r50_ddetr_dt_trainval_joint_ch.yaml --exp-name r50_ddetr_dt_trainval_joint_ch

################# old training config (only works on one node)
# echo "running script"
# python -m torch.distributed.run --nproc_per_node=4 main.py \
#                     --mode train \
#                     --config ./configs/r50_ddetr_dt_train.yaml \
#                     --use-wandb False \
#                     --outputs-per-step 10 \
#                     --exp-name r50_ddetr_dt_train_with_data_on_grete_2_nodes_more_cpu_per_task
#                     # --outputs-dir ./outputs/half_lr_compared_to_other_training \
#                     # --lr 5e-5 \
#                     # --resume-model /user/henrich1/u12041/repos/MOTIP/outputs/r50_deformable_detr_motip_dancetrack/checkpoint_7.pth