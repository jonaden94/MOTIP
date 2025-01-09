# Copyright (c) RuopengGao. All Rights Reserved.
import os
import torch.distributed
from utils.utils import (yaml_to_dict, is_main_process, set_seed,
                         init_distributed_mode, parse_option)
from log.logger import Logger
from configs.utils import update_config, load_super_config
from engines.train_engine import train
from engines.inference_engine import submit
from engines.inference_video_engine import video_info


def main(config: dict):
    """
    Main function.

    Args:
        config: Model configs.
    """
    # defining available GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = config["AVAILABLE_GPUS"]   # setting available gpus, like: "0,1,2,3"
    
    # init distributed mode depending on environment variables
    init_distributed_mode(config)

    # set directory where to save outputs
    config["OUTPUTS_DIR"] = os.path.join("./outputs", config["EXP_NAME"])
    if config["MODE"] == "train":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"])
    elif config["MODE"] == "inference":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["INFERENCE_SPLIT"], config["INFERENCE_MODEL"].split("/")[-1][:-4])
    elif config["MODE"] == "video_inference":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["VIDEO_DIR"].split("/")[-1], config["INFERENCE_MODEL"].split("/")[-1][:-4])
    else:
        raise NotImplementedError(f"Do not support running mode '{config['MODE']}' yet.")

    logger = Logger(
        logdir=log_dir,
        use_tensorboard=config["USE_TENSORBOARD"],
        use_wandb=config["USE_WANDB"],
        only_main=True,
        config=config
    )
    # Log runtime config.
    if is_main_process():
        logger.print_config(config=config, prompt="Runtime Configs: ")
        logger.save_config(config=config, filename="config.yaml")
        # logger.show(log=config, prompt="Main configs: ")
        # logger.write(config, "config.yaml")

    # set seed
    set_seed(config["SEED"])
    # Set num of CPUs
    if "NUM_CPU_PER_GPU" in config and config["NUM_CPU_PER_GPU"] is not None:
        torch.set_num_threads(config["NUM_CPU_PER_GPU"])

    if config["MODE"] == "train":
        train(config=config, logger=logger)
    elif config["MODE"] == "inference":
        submit(config=config, logger=logger)
    elif config["MODE"] == "video_inference":
        video_info(config=config, logger=logger)
    return


if __name__ == '__main__':
    opt = parse_option() # runtime options, a subset of .yaml config file (dict).
    cfg = yaml_to_dict(opt.config) # configs from .yaml file, path is set by runtime options.

    if opt.super_config_path is not None:
        cfg = load_super_config(cfg, opt.super_config_path)
    else:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Then, update configs by runtime options, using the different runtime setting.
    main(config=update_config(config=cfg, option=opt))
