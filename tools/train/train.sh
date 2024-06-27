python -m torch.distributed.run --nproc_per_node=2 main.py --mode train --use-distributed False --use-wandb False --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml --data-root ./datasets/ --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack/



python main.py --mode train --use-distributed False --use-wandb False --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml --data-root ./datasets/ --outputs-dir ./outputs/r50_deformable_detr_motip_dancetrack/