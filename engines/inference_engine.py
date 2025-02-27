# Copyright (c) RuopengGao. All Rights Reserved.
# About:
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from data.seq_dataset import SeqDataset
from data.video_dataset import VideoDataset
from utils.nested_tensor import tensor_list_to_nested_tensor
from models.utils import get_model
from utils.box_ops import box_cxcywh_to_xyxy
from collections import deque
from structures.instances import Instances
from structures.ordered_set import OrderedSet
from log.logger import Logger
from utils.utils import is_distributed, distributed_rank, distributed_world_size
from models import build_model
from models.utils import load_checkpoint
import cv2
import time
import numpy as np


def submit(config: dict, logger: Logger):
    """
    Submit a model for a specific dataset.
    :param config:
    :param logger:
    :return:
    """
    start_time = time.time()
    model = build_model(config)
    load_checkpoint(model, path=config["INFERENCE_MODEL"])

    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()])

    submit_outputs_dir = os.path.join(config["LOG_DIR"], 'results')

    # 需要调度整个 submit 流程
    submit_one_epoch(
        config=config,
        model=model,
        dataset=config["INFERENCE_DATASET"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=submit_outputs_dir,
        only_detr=config["INFERENCE_ONLY_DETR"]
    )

    logger.print(log=f"Finish inference with checkpoint '{config['INFERENCE_MODEL']}' on the {config['INFERENCE_DATASET']} {config['INFERENCE_SPLIT']} set, outputs are written to '{os.path.join(submit_outputs_dir)}/.")
    logger.save_log_to_file(
        log=f"Finish inference with checkpoint '{config['INFERENCE_MODEL']}' on the {config['INFERENCE_DATASET']} {config['INFERENCE_SPLIT']} set, outputs are written to '{os.path.join(submit_outputs_dir)}/.",
    )
    
    # Print the total inference time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    inference_time = f'{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}'
    logger.save_log_to_file(
        log=f"Total inference time: {inference_time}",
    )
    return


@torch.no_grad()
def submit_one_epoch(config: dict, model: nn.Module,
                     dataset: str, data_split: str,
                     outputs_dir: str, only_detr: bool = False):
    model.eval()
    
    if config['MODE'] == 'inference':
        inference_data_dir = os.path.join(config["DATA_ROOT"], dataset, data_split)
        all_seq_names = sorted(os.listdir(inference_data_dir))
        seq_names = [all_seq_names[_] for _ in range(len(all_seq_names))
                    if _ % distributed_world_size() == distributed_rank()]
        seq_paths = [os.path.join(inference_data_dir, _) for _ in seq_names]
    elif config['MODE'] == 'video_inference':
        for video in os.listdir(config["VIDEO_DIR"]):
            assert video.endswith(('.mp4', '.mkv')), f"Make sure all files in {config['VIDEO_DIR']} are either .mp4 or .mkv"
        all_seq_names = sorted(os.listdir(config["VIDEO_DIR"]))
        seq_names = [all_seq_names[_] for _ in range(len(all_seq_names))
                    if _ % distributed_world_size() == distributed_rank()]
        seq_paths = [os.path.join(config["VIDEO_DIR"], _) for _ in seq_names]
    else:
        raise NotImplementedError(f"Do not support running mode '{config['MODE']}'.")
        
    for seq_path in seq_paths:
        submit_one_seq(
            mode=config["MODE"],
            model=model,
            seq_dir=seq_path,
            only_detr=only_detr, max_temporal_length=config["MAX_TEMPORAL_LENGTH"],
            outputs_dir=outputs_dir,
            det_thresh=config["DET_THRESH"],
            newborn_thresh=config["DET_THRESH"] if "NEWBORN_THRESH" not in config else config["NEWBORN_THRESH"],
            area_thresh=config["AREA_THRESH"], id_thresh=config["ID_THRESH"],
            image_max_size=config["INFERENCE_MAX_SIZE"] if "INFERENCE_MAX_SIZE" in config else 1333,
            inference_ensemble=config["INFERENCE_ENSEMBLE"] if "INFERENCE_ENSEMBLE" in config else 0,
            draw_res=config["VISUALIZE_INFERENCE"]
        )

    if is_distributed():
        torch.distributed.barrier()
    return


@torch.no_grad()
def submit_one_seq(
            mode: str,
            model: nn.Module, seq_dir: str, outputs_dir: str,
            only_detr: bool, max_temporal_length: int = 0,
            det_thresh: float = 0.5, newborn_thresh: float = 0.5, area_thresh: float = 100, id_thresh: float = 0.1,
            image_max_size: int = 1333,
            inference_ensemble: int = 0,
            draw_res: bool = False
        ):
    # create output directories and paths
    tracker_dir = os.path.join(outputs_dir, "tracker")
    visualization_dir = os.path.join(outputs_dir, "visualization")
    os.makedirs(tracker_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    seq_name = os.path.split(seq_dir)[-1]
    if mode == "video_inference":
        seq_name = seq_name[:-4]
    
    video_w = None
    colors = (np.random.rand(32, 3) * 255).astype(dtype=np.int32)
    save_video_path = os.path.join(visualization_dir, f"{seq_name}.mp4")
    
    # instantiate dataset and other objects for getting tracks
    if mode == "inference":
        dataset = SeqDataset(seq_dir=seq_dir, width=image_max_size)
    elif mode == "video_inference":
        dataset = VideoDataset(video_path=seq_dir, width=image_max_size)
    else:
        raise NotImplementedError(f"Do not support running mode '{mode}'.")
    device = model.device
    current_id = 0
    ids_to_results = {}
    id_deque = OrderedSet() # an ID deque for inference, the ID will be recycled if the dictionary is not enough.

    # Trajectory history:
    if only_detr:
        trajectory_history = None
    else:
        trajectory_history = deque(maxlen=max_temporal_length)

    print(f"Start >> Submit seq {seq_name.split('/')[-1]}, {len(dataset)} frames ......")

    for i in range(dataset.__len__()):
        image, ori_image = dataset.__getitem__(i)
        ori_h, ori_w = ori_image.shape[0], ori_image.shape[1]
        frame = tensor_list_to_nested_tensor([image]).to(device)
        detr_outputs = model(frames=frame)
        detr_logits = detr_outputs["pred_logits"]
        detr_scores = torch.max(detr_logits, dim=-1).values.sigmoid()
        detr_det_idxs = detr_scores > det_thresh        # filter by the detection threshold
        detr_det_logits = detr_logits[detr_det_idxs]
        detr_det_labels = torch.max(detr_det_logits, dim=-1).indices
        detr_det_boxes = detr_outputs["pred_boxes"][detr_det_idxs]
        detr_det_outputs = detr_outputs["outputs"][detr_det_idxs]   # detr output embeddings
        area_legal_idxs = (detr_det_boxes[:, 2] * ori_w * detr_det_boxes[:, 3] * ori_h) > area_thresh   # filter by area
        detr_det_outputs = detr_det_outputs[area_legal_idxs]
        detr_det_boxes = detr_det_boxes[area_legal_idxs]
        detr_det_logits = detr_det_logits[area_legal_idxs]
        detr_det_labels = detr_det_labels[area_legal_idxs]

        # De-normalize to target image size:
        box_results = detr_det_boxes.cpu() * torch.tensor([ori_w, ori_h, ori_w, ori_h])
        box_results = box_cxcywh_to_xyxy(boxes=box_results)

        if only_detr is False:
            if len(box_results) > get_model(model).num_id_vocabulary:
                print(f"[Carefully!] we only support {get_model(model).num_id_vocabulary} ids, "
                      f"but get {len(box_results)} detections in seq {seq_name.split('/')[-1]} {i+1}th frame.")

        # Decoding the current objects' IDs
        if only_detr is False:
            assert max_temporal_length - 1 > 0, f"MOTIP need at least T=1 trajectory history, " \
                                                f"but get T={max_temporal_length - 1} history in Eval setting."
            current_tracks = Instances(image_size=(0, 0))
            current_tracks.boxes = detr_det_boxes
            current_tracks.outputs = detr_det_outputs
            current_tracks.ids = torch.tensor([get_model(model).num_id_vocabulary] * len(current_tracks),
                                              dtype=torch.long, device=current_tracks.outputs.device)
            current_tracks.confs = detr_det_logits.sigmoid()
            trajectory_history.append(current_tracks)
            if len(trajectory_history) == 1:    # first frame, do not need decoding:
                newborn_filter = (trajectory_history[0].confs > newborn_thresh).reshape(-1, )   # filter by newborn
                trajectory_history[0] = trajectory_history[0][newborn_filter]
                box_results = box_results[newborn_filter.cpu()]
                ids = torch.tensor([current_id + _ for _ in range(len(trajectory_history[-1]))],
                                   dtype=torch.long, device=current_tracks.outputs.device)
                trajectory_history[-1].ids = ids
                for _ in ids:
                    ids_to_results[_.item()] = current_id
                    current_id += 1
                id_results = []
                for _ in ids:
                    id_results.append(ids_to_results[_.item()])
                    id_deque.add(_.item())
                id_results = torch.tensor(id_results, dtype=torch.long)
            else:
                ids, trajectory_history, ids_to_results, current_id, id_deque, boxes_keep = get_model(model).inference(
                    trajectory_history=trajectory_history,
                    num_id_vocabulary=get_model(model).num_id_vocabulary,
                    ids_to_results=ids_to_results,
                    current_id=current_id,
                    id_deque=id_deque,
                    id_thresh=id_thresh,
                    newborn_thresh=newborn_thresh,
                    inference_ensemble=inference_ensemble,
                )   # already update the trajectory history/ids_to_results/current_id/id_deque in this function
                id_results = []
                for _ in ids:
                    id_results.append(ids_to_results[_])
                id_results = torch.tensor(id_results, dtype=torch.long)
                if boxes_keep is not None:
                    box_results = box_results[boxes_keep.cpu()]
        else:   # only detr, ID is just +1 for each detection.
            id_results = torch.tensor([current_id + _ for _ in range(len(box_results))], dtype=torch.long)
            current_id += len(id_results)

        # Write the outputs to the tracker file:
        result_file_path = os.path.join(tracker_dir, f"{seq_name}.txt")
        with open(result_file_path, "a") as file:
            assert len(id_results) == len(box_results), f"Boxes and IDs should in the same length, " \
                                                        f"but get len(IDs)={len(id_results)} and " \
                                                        f"len(Boxes)={len(box_results)}"
            for obj_id, box in zip(id_results, box_results):
                obj_id = int(obj_id.item())
                x1, y1, x2, y2 = box.tolist()
                result_line = f"{i + 1}," \
                                f"{obj_id}," \
                                f"{x1},{y1},{x2 - x1},{y2 - y1},1,-1,-1,-1\n"
                file.write(result_line)
                if draw_res:
                    color = tuple(colors[obj_id%32].tolist())
                    cv2.rectangle(ori_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(ori_image, str(obj_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, color, 2, cv2.LINE_AA)
            if draw_res:
                if video_w is None:
                    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    size = (ori_w, ori_h)
                    fps = 25
                    video_w = cv2.VideoWriter(save_video_path, fourcc, fps, size)
                    video_w.write(ori_image)
                else:
                    video_w.write(ori_image)
    if video_w is not None:
        video_w.release()
    print(f"Finish >> Submit seq {seq_name.split('/')[-1]}. ")
    return
