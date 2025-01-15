# Copyright (c) RuopengGao. All Rights Reserved.
import os
import torch.distributed
from utils.utils import (yaml_to_dict, is_main_process, set_seed,
                         init_distributed_mode, parse_option, munch_to_dict)
from log.logger import Logger
from configs.utils import update_config, load_super_config
from engines.train_engine import train
from engines.inference_engine import submit
import pprint


def main(config: dict):
    """
    Main function.

    Args:
        config: Model configs.
    """
    # init distributed mode depending on environment variables
    init_distributed_mode(config)

    # set directory where to save outputs
    config["OUTPUTS_DIR"] = os.path.join("./outputs", config["EXP_NAME"])
    if config["MODE"] == "train":
        config["LOG_DIR"] = os.path.join(config["OUTPUTS_DIR"], config["MODE"])
    elif config["MODE"] == "inference":
        config["LOG_DIR"] = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["INFERENCE_SPLIT"], config["INFERENCE_MODEL"].split("/")[-1][:-4])
    elif config["MODE"] == "video_inference":
        config["LOG_DIR"] = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["VIDEO_DIR"].split("/")[-1], config["INFERENCE_MODEL"].split("/")[-1][:-4])
    else:
        raise NotImplementedError(f"Do not support running mode '{config['MODE']}'.")

    logger = Logger(
        logdir=config["LOG_DIR"],
        use_tensorboard=config["USE_TENSORBOARD"],
        use_wandb=config["USE_WANDB"],
        only_main=True,
        config=config
    )
    # Log runtime config.
    if is_main_process():
        logger.print_config(config=config, prompt="Runtime Configs: ")
        logger.save_config(config=config, filename="config.yaml")
        logger.save_log_to_file(pprint.pformat(munch_to_dict(config)) + '\n\n')

    # set seed
    set_seed(config["SEED"])
    # Set num of CPUs
    if "NUM_CPU_PER_GPU" in config and config["NUM_CPU_PER_GPU"] is not None:
        torch.set_num_threads(config["NUM_CPU_PER_GPU"])

    if config["MODE"] == "train":
        train(config=config, logger=logger)
    elif config["MODE"] == "inference" or config["MODE"] == "video_inference":
        submit(config=config, logger=logger)
    else:
        raise NotImplementedError(f"Do not support running mode '{config['MODE']}'.")
    return


if __name__ == '__main__':
    opt = parse_option()
    cfg = yaml_to_dict(opt.config)
    cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    cfg = update_config(config=cfg, option=opt)
    main(config=cfg)
