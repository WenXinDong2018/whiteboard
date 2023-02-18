import numpy as np
import torch.nn as nn
import torch

# from dataclasses import dataclass


# @dataclass
# class Frame:
#     frame: torch.Tensor
#     time: int
#     average_pooled: torch.Tensor = 0.0

# Frames for every 2**i frames
frame_counters = np.array([0, 0, 0, 0, 0, 0])

frames = [None] * len(frame_counters)
num_frames = 3
k = 10
frames_average_pooled = [None] * len(frame_counters)


def process_frame(frame):
    global frames, frames_average_pooled, frame_counters

    frame_t = torch.tensor(frame, dtype=float).permute(2,0,1)
    C, H, W = frame_t.shape

    frame_counters += 1
    for i, count in enumerate(frame_counters):
        frame_counters[i] = count % (2**i)
        if frame_counters[i] == 0:
            frames[i] = frame_t
            frames_average_pooled[i] = nn.AvgPool2d((k,k), stride = (k,k))(frame_t)


    # average_pool = nn.AvgPool2d((k,k), stride = (k,k))(frame_t)
    # frames_average_pooled.append(average_pool)

    if any(frame is None for frame in frames):
        return frame

    # if len(frames)>num_frames:
    #     frames = frames[1:]
    #     frames_average_pooled = frames_average_pooled[1:]

    # curr_t = (num_frames-1)//2
    curr_average_pooled_frame = frames_average_pooled[0]

    average_delta = torch.abs(curr_average_pooled_frame - torch.stack(frames_average_pooled)).sum(axis=1).mean(axis=0)/3 #(N, C, H, W) => (N, H, W) => (H, W)
    mask = average_delta > 2
    mask = nn.Upsample((H, W))(mask.unsqueeze(0).unsqueeze(0)+0.0).squeeze()

    frames[0][:,mask>0] = 255
    ret = frames[0]
    return ret.permute(1,2,0).numpy().astype(np.uint8)
