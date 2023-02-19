import numpy as np
import torch.nn as nn
import torch
import copy
from collections import defaultdict
from dataclasses import dataclass
from scipy.ndimage.measurements import label

@dataclass
class Code:
    code: torch.Tensor
    board_freq: int
    obstacle_freq: int
    idx: int

class FrameBuffer:
    def __init__(self, num_frames: int, kernel_size: int, is_log_buffer=False, num_future_frames: int = 0) -> None:
        self.frame_buffer = []
        self.frames_average_pooled_buffer = []
        self.k = kernel_size
        self.num_frames = num_frames
        self.cur_frame = num_frames - num_future_frames - 1
        self.is_log_buffer = is_log_buffer
        self.avg_pool2d = nn.AvgPool2d((self.k,self.k), stride = (self.k//2,self.k//2))
        self.committed_frame = None
        # self.codebook = defaultdict(tuple)  # A list of Code
        # self.codebook_distance_eps = 10
        self.similarity_eps = 2
        self.t = 0
        self.foreground_color = torch.zeros((3,1))
        self.background_color = torch.zeros((3,1))

    def process_frame(self, frame):
        """
        Adds the frame to buffer and masks frame based on movement from previous frames.

        Args:
            frame (NDArray): Raw frame from video feed.

        Returns:
            masked_frame: Frames with movement masked out.
        """
        frame_tensor = torch.tensor(frame, dtype=float).permute(2,0,1) # type: ignore
        C, H, W = frame_tensor.shape
        self.frame_buffer.append(frame_tensor)
        self.frames_average_pooled_buffer.append(self.avg_pool2d(frame_tensor))

        if (self.is_log_buffer and len(self.frame_buffer) < 2**self.num_frames) or (
            not self.is_log_buffer and len(self.frame_buffer) < self.num_frames
        ):
            commited_frame, final_frame = self.commit_frame(None)
            return frame, commited_frame, final_frame

        # Pop oldest frame
        self.frame_buffer.pop(0)
        self.frames_average_pooled_buffer.pop(0)

        current_frame_avg_pooled = self.frames_average_pooled_buffer[self.cur_frame]
        if self.is_log_buffer:
            indexes = list(-1 - 2 ** i for i in range(self.num_frames))
            average_delta = torch.abs(current_frame_avg_pooled - torch.stack(tuple(self.frames_average_pooled_buffer[i] for i in indexes))).sum(axis=1).mean(axis=0)/3 # type: ignore
        else:
            average_delta = torch.abs(current_frame_avg_pooled - torch.stack(self.frames_average_pooled_buffer)).sum(axis=1).mean(axis=0)/3 # type: ignore #(N, C, H, W) => (N, H, W) => (H, W)

        mask = average_delta > 2

        self._fill_mask_gaps(mask)

        mask = nn.Upsample((H, W))(mask.unsqueeze(0).unsqueeze(0)+0.0).squeeze()
        final_frame, commited_frame = self.commit_frame(mask>0)

        self.frame_buffer[self.cur_frame][:,mask>0] = -1
        processed_frame = self.to_cv_frame(self.frame_buffer[self.cur_frame])

        return processed_frame, commited_frame, final_frame

    def _fill_mask_gaps(self, mask):

        structure = np.ones((3, 3), dtype=np.int)

        # Get rid of blinking noises
        labeled, ncomponents = label(mask, structure)
        labeled = np.array(labeled)

        for i in range(1, ncomponents):
            component_size = np.sum(labeled == i)
            if component_size < 15:
                # print(f"component with size {component_size} is background")
                #component is background
                mask[labeled==i] = 0

        # Fill obstacles
        labeled, ncomponents = label(torch.logical_not(mask), structure)
        max_component_size = 0
        for i in range(1, ncomponents):
            component_size = np.sum(labeled == i)
            max_component_size = max(max_component_size, component_size)

        for i in range(1,ncomponents):
            component_size = np.sum(labeled == i)
            if component_size < max_component_size:
                #component is obstacle
                # print(f"component with size {component_size} is obstacle")
                mask[labeled==i] = 1


    def to_cv_frame(self, frame):
        """
        Converts a tensorflow frame (channel, height, width) to a cv2 frame (height, width, channel)
        """
        return frame.permute(1,2,0).numpy().astype(np.uint8)

    def moving_average(self,color, new_colors):
        if not len(new_colors): return color
        new_color = torch.mean(new_colors, axis=1).unsqueeze(1) #length 3 vector
        return (self.t*color + new_color)/(self.t+1)


    def commit_frame(self, mask): #mask is 1 for obstacles
        """
        Commits current frame to be used as reference frame for future frames.
        """

        if self.committed_frame is None: #Commit the very first frame
            self.committed_frame = copy.deepcopy(self.frame_buffer[0])
            return self.to_cv_frame(self.committed_frame), self.to_cv_frame(self.frame_buffer[0])
        if mask==None: #Assume the first frame is obstacle-free
            return self.to_cv_frame(self.committed_frame), self.to_cv_frame(self.frame_buffer[0])

        # self.cur_frame = self.num_frames//2

        foreground = self.frame_buffer[self.cur_frame][:, mask]
        background = self.frame_buffer[self.cur_frame][:, torch.logical_not(mask)]

        self.foreground_color = self.moving_average(self.foreground_color, foreground)
        self.background_color = self.moving_average(self.background_color, background)

        difference = torch.abs(self.committed_frame - self.frame_buffer[self.cur_frame]).sum(axis=0)/3
        distance_to_foreground = torch.abs(self.foreground_color.unsqueeze(-1) - self.frame_buffer[self.cur_frame]).sum(axis=0)/3 + 1
        distance_to_background = torch.abs(self.background_color.unsqueeze(-1) - self.frame_buffer[self.cur_frame]).sum(axis=0)/3 + 1
        ratio = distance_to_background / distance_to_foreground

        commited = torch.logical_and(torch.logical_not(mask), torch.logical_or(ratio<1, difference<10))
        commited = torch.logical_not(mask)
        # Commiting the background
        self.committed_frame[:,commited] = self.frame_buffer[self.cur_frame][:,commited]
        # Add semi-transparent obstacles
        not_commited = torch.logical_not(commited)
        show_frame = copy.deepcopy(self.committed_frame)
        beta = 5
        show_frame[:, not_commited] = show_frame[:, not_commited] *(beta-1)/beta + self.frame_buffer[self.cur_frame][:, not_commited]*1/beta
        mask = not_commited
        return self.to_cv_frame(show_frame), self.to_cv_frame(self.committed_frame)

