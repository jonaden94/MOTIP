# not distributed
# python main.py --mode train --use-distributed False --use-wandb False --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml --data-root ./datasets/ --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack/

# standard config
# python -m torch.distributed.run --nproc_per_node=4 main.py --mode train --use-distributed True --use-wandb False --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml --data-root ./datasets/ --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack/


# lower learning rate
# python -m torch.distributed.run --nproc_per_node=4 main.py --mode train --use-distributed True --use-wandb False --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml --data-root ./datasets/ --outputs-dir ./outputs/half_lr_compared_to_other_training --lr 5e-5