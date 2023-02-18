import numpy as np
import torch.nn as nn
import torch

# from dataclasses import dataclass


# @dataclass
# class Frame:
#     frame: torch.Tensor
#     time: int
#     average_pooled: torch.Tensor = 0.0


frames = []
num_frames = 11 * 2 + 1
k = 10
frames_average_pooled = []


def process_frame(frame):
    global frames, frames_average_pooled

    frame_t = torch.tensor(frame, dtype=float).permute(2, 0, 1)
    C, H, W = frame_t.shape
    frames.append(frame_t)

    average_pool = nn.AvgPool2d((k, k), stride=(k, k))(frame_t)
    frames_average_pooled.append(average_pool)

    if len(frames) < num_frames:
        return frame

    if len(frames) > num_frames:
        frames = frames[1:]
        frames_average_pooled = frames_average_pooled[1:]

    curr_t = (num_frames - 1) // 2
    curr_average_pooled_frame = frames_average_pooled[curr_t]

    average_delta = torch.abs(curr_average_pooled_frame - torch.stack(frames_average_pooled)).sum(axis=1).mean(
        axis=0) / 3  # (N, C, H, W) => (N, H, W) => (H, W)
    mask = average_delta > 2
    mask = nn.Upsample((H, W))(mask.unsqueeze(0).unsqueeze(0) + 0.0).squeeze()

    frames[curr_t][:, mask > 0] = 255
    ret = frames[curr_t]
    return ret.permute(1, 2, 0).numpy().astype(np.uint8)
