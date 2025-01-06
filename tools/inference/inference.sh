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
