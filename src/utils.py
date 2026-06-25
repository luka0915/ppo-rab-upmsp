import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_config(cfg, dataset):
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    with open(f"{cfg.MODEL_DIR}config_settings.txt", 'w') as f:
        f.write("PPO-RAB Training Configuration\n")
        f.write("=" * 50 + "\n\n")

        f.write("Seeds:\n")
        f.write(f"TRAIN_SEED: {cfg.TRAIN_SEED}\n")
        f.write(f"TEST_SEED: {cfg.TEST_SEED}\n\n")

        f.write("PPO hyperparameters:\n")
        f.write(f"NUM_EPISODES (N_E): {cfg.NUM_EPISODES}\n")
        f.write(f"GAMMA: {cfg.GAMMA}\n")
        f.write(f"GAE_LAMBDA: {cfg.GAE_LAMBDA}\n")
        f.write(f"ACTOR_LR: {cfg.ACTOR_LR}\n")
        f.write(f"CRITIC_LR: {cfg.CRITIC_LR}\n")
        f.write(f"BATCH_SIZE (N_B): {cfg.BATCH_SIZE}\n")
        f.write(f"NUM_MINI_BATCHES: {cfg.NUM_MINI_BATCHES}\n")
        f.write(f"UPDATE_EPOCHS (N_U): {cfg.UPDATE_EPOCHS}\n")
        f.write(f"EPSILON: {cfg.EPSILON}\n")
        f.write(f"TEMPERATURE (tau): {getattr(cfg, 'TEMPERATURE', 0.1)}\n")
        f.write(f"ETA: {cfg.ETA}\n")
        f.write(f"MAX_GRAD_NORM: {cfg.MAX_GRAD_NORM}\n")
        f.write(f"TIME_LIMIT: {cfg.TIME_LIMIT}\n")
