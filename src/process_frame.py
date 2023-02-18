import numpy as np
import torch.nn as nn
import torch

# from dataclasses import dataclass


# @dataclass
# class Frame:
#     frame: torch.Tensor
#     time: int
#     average_pooled: torch.Tensor = 0.0

class FrameBuffer:
    def __init__(self, num_frames: int, kernel_size: int, is_log_buffer=False) -> None:
        self.frame_buffer = []
        self.frames_average_pooled_buffer = []
        self.k = kernel_size
        self.num_frames = num_frames
        self.is_log_buffer = is_log_buffer
        self.avg_pool2d = nn.AvgPool2d((self.k,self.k), stride = (self.k,self.k))

    def process_frame(self, frame):
        frame_tensor = torch.tensor(frame, dtype=float).permute(2,0,1) # type: ignore
        self.C, self.H, self.W = frame_tensor.shape
        self.frame_buffer.append(frame_tensor)
        self.frames_average_pooled_buffer.append(
            self.avg_pool2d(frame_tensor)
        )

        if ((self.is_log_buffer and len(self.frame_buffer) < 2**self.num_frames) or
                (not self.is_log_buffer and len(self.frame_buffer) < self.num_frames)):
            return frame

        # Pop oldest frame
        self.frame_buffer.pop(0)
        self.frames_average_pooled_buffer.pop(0)

        current_frame_avg_pooled = self.frames_average_pooled_buffer[-1]
        if self.is_log_buffer:
            indexes = list(-1 - 2 ** i for i in range(self.num_frames))
            average_delta = torch.abs(current_frame_avg_pooled - torch.stack(tuple(self.frames_average_pooled_buffer[i] for i in indexes))).sum(axis=1).mean(axis=0)/3 # type: ignore
        else:
            average_delta = torch.abs(current_frame_avg_pooled - torch.stack(self.frames_average_pooled_buffer)).sum(axis=1).mean(axis=0)/3 # type: ignore #(N, C, H, W) => (N, H, W) => (H, W)

        mask = average_delta > 2
        mask = nn.Upsample((self.H, self.W))(mask.unsqueeze(0).unsqueeze(0)+0.0).squeeze()

        self.frame_buffer[-1][:,mask>0] = -1
        ret = self.frame_buffer[-1]
        return ret.permute(1,2,0).numpy().astype(np.uint8)


# Frames for every 2**i frames
frames = []
num_frames = 3
k = 10
frames_average_pooled = []


def process_frame(frame):
    print('Deprecated function. Use FrameBuffer class instead.')
    global frames, frames_average_pooled, frame_counters

    frame_t = torch.tensor(frame, dtype=float).permute(2,0,1) # type: ignore
    C, H, W = frame_t.shape

    frames.append(frame_t)

    average_pool = nn.AvgPool2d((k,k), stride = (k,k))(frame_t)
    frames_average_pooled.append(average_pool)

    if len(frames)<num_frames:
        return frame

    if len(frames)>num_frames:
        frames = frames[1:]
        frames_average_pooled = frames_average_pooled[1:]

    curr_t = (num_frames-1)
    curr_average_pooled_frame = frames_average_pooled[curr_t]

    average_delta = torch.abs(curr_average_pooled_frame - torch.stack(frames_average_pooled)).sum(axis=1).mean(axis=0)/3 # type: ignore #(N, C, H, W) => (N, H, W) => (H, W)
    mask = average_delta > 2
    mask = nn.Upsample((H, W))(mask.unsqueeze(0).unsqueeze(0)+0.0).squeeze()

    frames[curr_t][:,mask>0] = 255
    ret = frames[curr_t]
    return ret.permute(1,2,0).numpy().astype(np.uint8)
