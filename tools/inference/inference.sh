#!/bin/bash
#SBATCH -p grete
#SBATCH --nodes=2                # node count
#SBATCH --gpus-per-node=A100:4   # total number of gpus per node
#SBATCH --ntasks-per-node=4      # total number of tasks per node
##SBATCH --nodes=1                # node count
##SBATCH --gpus-per-node=A100:1   # total number of gpus per node
##SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -A nib00034
#SBATCH -C inet
#SBATCH -o /user/henrich1/u12041/output/job-%J.out
#SBATCH --mem=256G
#SBATCH -t 0-05:00:00

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
export WANDB_DISABLE_GIT=true

echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$SLURM_NTASKS
echo "MASTER_ADDR="$MASTER_ADDR

srun python main.py \
    --mode inference \
    --use-wandb False \
    --config ./configs/r50_ddetr_dt_train.yaml \
    --exp-name r50_ddetr_dt_train \
    --inference-model ./outputs/r50_ddetr_dt_train/checkpoint12.pth \
    --inference-dataset DanceTrack \
    --inference-split val \
    --visualize-inference True

####################################### inference with torch.distributed.run (only works on one node for me) 
# python -m torch.distributed.run --nproc_per_node=4 main.py \
#                                 --mode inference \
#                                 --use-wandb False \
#                                 --config ./configs/r50_ddetr_dt_train.yaml \
#                                 --exp-name r50_ddetr_dt_train \
#                                 --inference-model ./outputs/r50_ddetr_dt_train/checkpoint_12.pth \
#                                 --inference-dataset DanceTrack \
#                                 --inference-split test
















# #!/bin/bash TODO
# #SBATCH -p grete
# #SBATCH -G A100:4

# #SBATCH -t 2-00:00:00
# #SBATCH -A nib00034
# #SBATCH -C inet
# #SBATCH -o /user/henrich1/u12041/output/job-%J.out
# #SBATCH --mem=256G
# ##SBATCH --mail-type=begin            # send mail when job begins
# ##SBATCH --mail-type=end              # send mail when job ends
# ##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
# export HTTP_PROXY="http://www-cache.gwdg.de:3128"

# echo "Activating conda..."
# CONDA_BASE=$(conda info --base)
# echo $CONDA_BASE
# source $CONDA_BASE/etc/profile.d/conda.sh
# conda activate motip
# cd repos/MOTIP

# # normal lr
# echo "running script"
# python -m torch.distributed.run --nproc_per_node=4 main.py \
#                                 --mode submit \
#                                 --use-distributed True \
#                                 --use-wandb False \
#                                 --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml \
#                                 --data-root ./datasets/ \
#                                 --inference-model ./outputs/r50_deformable_detr_motip_dancetrack/checkpoint_12.pth \
#                                 --outputs-dir ./outputs/dancetrack_trackers/ \
#                                 --inference-dataset DanceTrack \
#                                 --inference-split test
# echo "Python script executed"
