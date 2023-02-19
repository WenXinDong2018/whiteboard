from __future__ import annotations

import cv2

import process_frame as pf
import time
from typing import Union

class ImageFeed:
    def __init__(self, video_input: Union[str, int],
                 frame_buffers: list[pf.FrameBuffer],
                 frame_buffer_names: list[str] | None = None) -> None:
        """
        Image feed class that wraps video capturing and processing.

        Args:
            video_input (Union[str, int]): Input string to path of video or 0 for webcam feed.
            frame_buffers (list[pf.FrameBuffer]): List of frame buffers to process frames with.
            frame_buffer_names (list[str]): List of names for each frame buffer, used for window management.
        """
        if frame_buffer_names is not None and len(frame_buffers) != len(frame_buffer_names):
            raise ValueError('Number of frame buffers and frame buffer names must be equal.')
        if isinstance(video_input, int) and video_input != 0:
            print('Invalid input for webcam feed. Defaulting to webcam feed.')
            video_input = 0
        self.windows: list[str] = []
        self.frame_buffers = frame_buffers
        if frame_buffer_names is None:
            frame_buffer_names = [f'frame buffer {i}' for i in range(len(frame_buffers))]
        self.frame_buffer_names = frame_buffer_names

    def _start_feed(self):
        self.vid = cv2.VideoCapture(input)

    def _stop_feed(self):
        self.vid.release()
        for win in self.windows:
            cv2.destroyWindow(win)

    def start_capture_loop(self):
        self._start_feed()
        try:
            while True:
                ret, frame = self.vid.read()
                if not ret:
                    break
                for frame_buffer in self.frame_buffers:
                    masked_frame = frame_buffer.process_frame(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except:
            print('Error in capture loop.')
        self._stop_feed()

video_input = "../videos/terry_tao_low_res.mp4"

vid = cv2.VideoCapture(0)

# frame_buffer_log = pf.FrameBuffer(6, 10, is_log_buffer=True)
frame_buffer_lin = pf.FrameBuffer(50, 10, is_log_buffer=False)

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    if not ret:
        break

    start = time.time()

    # Display the resulting frame
    # processed_frame_log = frame_buffer_log.process_frame(frame)
    processed_frame_lin = frame_buffer_lin.process_frame(frame)

    # commit_frame_log = frame_buffer_log.commit_frame()
    commit_frame_lin = frame_buffer_lin.commit_frame()
    # processed_frame = pf.process_frame(frame)
    # cv2.imshow('log frame processed', processed_frame_log)
    cv2.imshow('lin frame processed', processed_frame_lin)
    # cv2.imshow('log frame committed', commit_frame_log)
    cv2.imshow('lin frame committed', commit_frame_lin)

    end = time.time()

    print(f'Time for single iteration: {end - start}')
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()