from __future__ import annotations

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import os
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import torch
from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.visualize.visualize import viz_clip_text, get_gpu_device
from src.utils.misc import load_model_wo_clip
import src.utils.fixseed  # noqa

plt.switch_backend('agg')

def get_motion_clip_model():
    parameters, folder, checkpointname, epoch = parser()
    gpu_device = get_gpu_device()
    parameters["device"] = f"cuda:{gpu_device}"
    model, datasets = get_model_and_data(parameters, split='vald')
    return model


def encode_motions(model, motions, device):
    return model.encoder({'x': motions,
                          'y': torch.zeros(motions.shape[0], dtype=int, device=device),
                          'mask': model.lengths_to_mask(torch.ones(motions.shape[0], dtype=int, device=device) * 60)})["mu"]


class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "MotionCLIP"):
        super().__init__()
        assert name == "MotionCLIP"
        self.model = get_motion_clip_model()
        self.model.eval().requires_grad_(False)

    def encode_text(self, text: list[str]) -> torch.Tensor:
        return self.model.clip_model.encode_text(text).float()
    
    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        motions = motion.unsqueeze(0).permute((0, 2, 3, 1))
        return self.model.encoder({'x': motions,
                                   'y': torch.zeros(motions.shape[0], dtype=int, device=self.model.device),
                                   'mask': self.model.lengths_to_mask(torch.ones(motions.shape[0], dtype=int, device=self.model.device) * 60)})["mu"].float()


    def forward(
        self, motion_0: torch.Tensor, motion_1: torch.Tensor, text_0: list[str], text_1: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_features_0 = self.encode_motion(motion_0)
        motion_features_1 = self.encode_motion(motion_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim_0 = F.cosine_similarity(motion_features_0, text_features_0)
        sim_1 = F.cosine_similarity(motion_features_1, text_features_1)
        sim_direction = F.cosine_similarity(motion_features_1 - motion_features_0, text_features_1 - text_features_0)
        sim_image = F.cosine_similarity(motion_features_0, motion_features_1)
        return sim_0, sim_1, sim_direction, sim_image
    

if __name__ == '__main__':
    clip_similarity = ClipSimilarity()
    data_golf = np.load('/home/ctq566/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_p2p_trial/sample00_rep00.npy')
    data_tennis = np.load('/home/ctq566/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_p2p_trial/sample01_rep00.npy')

    # clip_similarity.encode_motion(torch.tensor(data))
    print(clip_similarity(torch.tensor(data_golf, device='cuda:0'), torch.tensor(data_tennis, device='cuda:0'), ['grabing a golf club and lightly moving it'], ['grabbing a golf club quickly and moving it briefly']))