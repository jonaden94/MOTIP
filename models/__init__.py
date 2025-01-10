# Copyright (c) RuopengGao. All Rights Reserved.
import torch
from .motip import build as build_motl

def build_model(config: dict):
    model = build_motl(config=config)
    model.to(device=torch.device(config["DEVICE"]))
    return model