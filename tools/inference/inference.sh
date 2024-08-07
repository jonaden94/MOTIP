python -m torch.distributed.run --nproc_per_node=4 main.py \
                                --mode submit \
                                --use-distributed True \
                                --use-wandb False \
                                --config-path ./configs/r50_deformable_detr_motip_dancetrack.yaml \
                                --data-root ./datasets/ \
                                --inference-model ./outputs/r50_deformable_detr_motip_dancetrack/checkpoint_12.pth \
                                --outputs-dir ./outputs/dancetrack_trackers/ \
                                --inference-dataset DanceTrack \
                                --inference-split test
