import numpy as np
import torch.nn as nn
import torch
import copy
from collections import defaultdict
from dataclasses import dataclass
from scipy.ndimage.measurements import label

frames = []
num_frames = 11 * 2 + 1
k = 50
frames_average_pooled = []

@dataclass
class Code:
    code: torch.Tensor
    board_freq: int
    obstacle_freq: int
    idx: int

class FrameBuffer:
    def __init__(self, num_frames: int, kernel_size: int, is_log_buffer=False) -> None:
        self.frame_buffer = []
        self.frames_average_pooled_buffer = []
        self.k = kernel_size
        self.num_frames = num_frames
        self.is_log_buffer = is_log_buffer
        self.avg_pool2d = nn.AvgPool2d((self.k,self.k), stride = (self.k//2,self.k//2))
        self.committed_frame = None
        self.codebook = defaultdict(tuple)  # A list of Code
        self.codebook_distance_eps = 10
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
        self.C, self.H, self.W = frame_tensor.shape
        self.frame_buffer.append(frame_tensor)
        self.frames_average_pooled_buffer.append(self.avg_pool2d(frame_tensor))

        if (self.is_log_buffer and len(self.frame_buffer) < 2**self.num_frames) or (
            not self.is_log_buffer and len(self.frame_buffer) < self.num_frames
        ):
            return frame, frame

        # Pop oldest frame
        self.frame_buffer.pop(0)
        self.frames_average_pooled_buffer.pop(0)

        current_frame_avg_pooled = self.frames_average_pooled_buffer[-1]
        if self.is_log_buffer:
            indexes = list(-1 - 2 ** i for i in range(self.num_frames))
            average_delta = torch.abs(current_frame_avg_pooled - torch.stack(tuple(self.frames_average_pooled_buffer[i] for i in indexes))).sum(axis=1).mean(axis=0)/3 # type: ignore
        else:
            average_delta = torch.abs(current_frame_avg_pooled - torch.stack(self.frames_average_pooled_buffer)).sum(axis=1).mean(axis=0)/3 # type: ignore #(N, C, H, W) => (N, H, W) => (H, W)

        mask = average_delta > 1

        # self._update_codebook(mask, current_frame_avg_pooled)
        self._fill_mask_gaps(mask)

        mask = nn.Upsample((self.H, self.W))(mask.unsqueeze(0).unsqueeze(0)+0.0).squeeze()
        commited_frame = self.commit_frame(mask>0)
        self.frame_buffer[-1][:,mask>0] = -1
        processed_frame = self.to_cv_frame(self.frame_buffer[-1])
        return processed_frame, commited_frame

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
        self.t += 1
        if self.committed_frame is None:
            self.committed_frame = self.frame_buffer[-1]
            return self.to_cv_frame(self.committed_frame)

        foreground = self.frame_buffer[-1][:, mask]
        background = self.frame_buffer[-1][:,torch.logical_not(mask)]

        self.foreground_color = self.moving_average(self.foreground_color, foreground)
        self.background_color = self.moving_average(self.background_color, background)

        difference = torch.abs(self.committed_frame - self.frame_buffer[-1]).sum(axis=0)/3
        # print("self.frame_buffer[-1][mask].reshape(3, -1)",self.frame_buffer[-1][mask].reshape(3, -1).shape)
        # print("self.frame_buffer[-1]", self.frame_buffer[-1].shape)
        distance_to_foreground = torch.abs(self.foreground_color.unsqueeze(-1) - self.frame_buffer[-1]).sum(axis=0)/3 + 1
        distance_to_background = torch.abs(self.background_color.unsqueeze(-1) - self.frame_buffer[-1]).sum(axis=0)/3 + 1
        # print("distance_to_background", distance_to_background.shape)
        ratio = distance_to_background / distance_to_foreground
        # print("ratio", ratio.shape)
        # print("ratio<2/ratio", torch.sum(ratio<2), len(ratio))
        # print("self.committed_frame[:,mask]", self.committed_frame[:,torch.logical_not(mask).nonzero()[ratio]].shape)
        commited = torch.logical_and(torch.logical_not(mask), torch.logical_or(ratio<0.7, difference<5))
        # Commiting the background
        self.committed_frame[:,commited] = self.frame_buffer[-1][:,commited]
        not_commited = torch.logical_not(commited)
        show_frame = copy.deepcopy(self.committed_frame)
        beta = 5
        show_frame[:, not_commited] = show_frame[:, not_commited]*(beta-1)/beta + self.frame_buffer[-1][:, not_commited]*1/beta
        mask = not_commited
        return self.to_cv_frame(show_frame)

    def _update_codebook(self, mask, average_pooled):
        # Randomly select indices of average_pooled to speed up
        C, H, W = average_pooled.shape
        assert(H == mask.shape[0])
        assert(W == mask.shape[1])
        for i in range(H):
            for j in range(W):
                code = average_pooled[:,i, j]
                match = self._find_closest_code(code, create = True)
                if mask[i][j]>0:
                    match.obstacle_freq +=1
                else:
                    match.board_freq += 1

    def _update_codebook(self, mask, average_pooled):
        # Randomly select indices of average_pooled to speed up
        C, H, W = average_pooled.shape
        assert H == mask.shape[0]
        assert W == mask.shape[1]
        for i in range(0,H,20):
            for j in range(0,W,20):
                code = average_pooled[:, i, j]
                match = self._find_closest_code(code, create=True)
                if mask[i][j] > 0:
                    match.obstacle_freq += 1
                else:
                    match.board_freq += 1

    def _most_likely_obstacle(self, code):
        return code.board_freq > 5 and code.obstacle_freq / code.board_freq > 2

    def _most_likely_board(self, code):
        return code.obstacle_freq > 5 and code.board_freq / code.obstacle_freq > 2

    def _find_closest_code(self, x, create=False):

        add_code = True
        min_dist = float("inf")
        closest_code = None
        target = (int(x[0]), int(x[1]), int(x[2]))

        for r in range(
            max(0, target[0] - self.codebook_distance_eps),
            min(255, target[0] + self.codebook_distance_eps),
        ):
            for g in range(
                max(0, target[1] - self.codebook_distance_eps),
                min(255, target[1] + self.codebook_distance_eps),
            ):
                for b in range(
                    max(0, target[2] - self.codebook_distance_eps),
                    min(255, target[2] + self.codebook_distance_eps),
                ):
                    if (r, g, b) in self.codebook:
                        add_code = False
                        code = self.codebook[(r, g, b)]
                        dist = torch.abs(code.code - x).sum() / 3
                        if dist < min_dist:
                            min_dist = dist
                            closest_code = code

        if create and add_code:
            self.codebook[target] = Code(x, 0, 0, len(self.codebook))
            closest_code = self.codebook[target]
        return closest_code

    def _update_mask(self, mask, average_pooled):
        C, H, W = average_pooled.shape
        for i in range(H):
            for j in range(W):
                code = average_pooled[:, i, j]
                match = self._find_closest_code(code, create=False)
                if match != None:
                    if self._most_likely_obstacle(match):
                        mask[i, j] = 1
                    elif self._most_likely_board(match):
                        mask[i, j] = 0
