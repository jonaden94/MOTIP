#!/bin/bash
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
export https_proxy="http://www-cache.gwdg.de:3128"
export http_proxy="http://www-cache.gwdg.de:3128"


echo "Activating conda..."
CONDA_BASE=$(conda info --base)
echo $CONDA_BASE
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate motip
cd ~/repos/MOTIP


echo "running script"
python -m torch.distributed.run --nproc_per_node=4 main.py \
                    --mode train \
                    --use-distributed True \
                    --config-path ./configs/pigs/r50_deformable_detr_motip_pigtrack_joint_pigdetect.yaml \
                    --data-root ./datasets/ \
                    --use-wandb True \
                    # --outputs-dir ./outputs/r50_deformable_detr_motip_pigtrack_joint_pigdetect \
                    # --resume-model /user/henrich1/u12041/repos/MOTIP/outputs/r50_deformable_detr_motip_dancetrack/checkpoint_7.pth
echo "Python script executed"
