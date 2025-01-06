# Copyright (c) RuopengGao. All Rights Reserved.
import os
import torch.distributed
from utils.utils import (yaml_to_dict, is_main_process, set_seed,
                         init_distributed_mode, parse_option)
from log.logger import Logger
from configs.utils import update_config, load_super_config
from train_engine import train
from eval_engine import evaluate
from submit_engine import submit
from video_infer_engine import video_info


def main(config: dict, video_path=None):
    """
    Main function.

    Args:
        config: Model configs.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = config["AVAILABLE_GPUS"]   # setting available gpus, like: "0,1,2,3"

    _not_use_tf32 = False if "USE_TF32" not in config else not config["USE_TF32"]
    if _not_use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if is_main_process():
            print("Not use TF32 on Ampere GPUs.")
            
    init_distributed_mode(config)

    # You can only set the `--exp-name` in runtime option,
    # if `--outputs-dir` is None, it will be set to `./outputs/[exp_name]/`.
    if config["OUTPUTS_DIR"] is None:
        config["OUTPUTS_DIR"] = os.path.join("./outputs", config["EXP_NAME"])

    if config["MODE"] == "train":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"])
    elif config["MODE"] == "eval":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"],
                               config["INFERENCE_GROUP"] if config["INFERENCE_GROUP"] is not None else "default",
                               config["INFERENCE_SPLIT"])
    elif config["MODE"] == "submit":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"],
                               config["INFERENCE_GROUP"] if config["INFERENCE_GROUP"] is not None else "default",
                               config["INFERENCE_SPLIT"])
    elif config["MODE"] == "video":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"])
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
    elif config["MODE"] == "eval":
        evaluate(config=config, logger=logger)
    elif config["MODE"] == "submit":
        submit(config=config, logger=logger)
    elif config["MODE"] == "video":
        if video_path is not None:
            video_info(video_path, config=config, logger=logger)
        else:
            print('Supply path to video file!')
    return


if __name__ == '__main__':
    opt = parse_option() # runtime options, a subset of .yaml config file (dict).
    cfg = yaml_to_dict(opt.config) # configs from .yaml file, path is set by runtime options.

    if opt.super_config_path is not None:
        cfg = load_super_config(cfg, opt.super_config_path)
    else:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Then, update configs by runtime options, using the different runtime setting.
    main(config=update_config(config=cfg, option=opt), video_path=opt.video_path)
