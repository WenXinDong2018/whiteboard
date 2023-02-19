from __future__ import annotations

import cv2

import process_frame as pf
import time
from typing import Union
import argparse

class ImageFeed:
    def __init__(self, video_input: Union[str, int],
                 frame_buffers: list[pf.FrameBuffer] | pf.FrameBuffer,
                 frame_buffer_names: list[str] | str | None = None,
                 verbose: bool = False) -> None:
        """
        Image feed class that wraps video capturing and processing.

        Args:
            video_input (Union[str, int]): Input string to path of video or 0 for webcam feed.
            frame_buffers (list[pf.FrameBuffer]): List of frame buffers to process frames with.
            frame_buffer_names (list[str]): List of names for each frame buffer, used for window management.
            verbose (bool): Whether to print verbose output.
        """
        # Frame buffer validation
        if isinstance(frame_buffers, pf.FrameBuffer) and isinstance(frame_buffer_names, list) and len(frame_buffer_names) != 1:
            raise ValueError('Frame buffer and frame buffer names must be both lists or FrameBuffer and str respectively.')
        elif isinstance(frame_buffers, list) and isinstance(frame_buffer_names, list):
            if frame_buffer_names is not None and len(frame_buffers) != len(frame_buffer_names):
                raise ValueError('Number of frame buffers and frame buffer names must be equal.')

        if isinstance(frame_buffers, pf.FrameBuffer):
            frame_buffers = [frame_buffers]
        if isinstance(frame_buffer_names, str):
            frame_buffer_names = [frame_buffer_names]

        if isinstance(video_input, int) and video_input != 0:
            print('Invalid input for webcam feed. Defaulting to webcam feed.')
            video_input = 0
        self.video_input = video_input
        self.windows: list[str] = []
        self.frame_buffers = frame_buffers
        if frame_buffer_names is None:
            frame_buffer_names = [f'frame buffer {i}' for i in range(len(frame_buffers))]
        self.frame_buffer_names = frame_buffer_names
        self.windows = [f'{frame_buffer_name} masked' for frame_buffer_name in frame_buffer_names] + \
            [f'{frame_buffer_name} committed' for frame_buffer_name in frame_buffer_names]
        self.verbose = verbose

    def _start_feed(self):
        self.vid_capture = cv2.VideoCapture(self.video_input)

    def _stop_feed(self):
        self.vid_capture.release()
        for win in self.windows:
            cv2.destroyWindow(win)

    def run_capture_loop(self):
        self._start_feed()
        i = 0
        avg_scope = 100
        start_time = time.time()
        global_start_time = start_time
        try:
            while True:
                i += 1
                if i % avg_scope == 0 and self.verbose:
                    print(f'Frames {i-avg_scope}-{i} average time spent: {(time.time() - start_time) / avg_scope}s per frame')
                    start_time = time.time()
                ret, frame = self.vid_capture.read()
                if not ret:
                    break
                for frame_buffer, frame_buffer_name in zip(self.frame_buffers, self.frame_buffer_names):
                    masked_frame, committed_frame = frame_buffer.process_frame(frame)
                    # committed_frame = frame_buffer.commit_frame()
                    cv2.imshow(f'{frame_buffer_name} masked', masked_frame)
                    cv2.imshow(f'{frame_buffer_name} committed', committed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(e)
            print('Error in capture loop.')
        self._stop_feed()

        end = time.time()
        print(f'Average time spent per frame: {(end - global_start_time) / i}s per frame')
        print(f'Average FPS: {i / (end - global_start_time)}')
        print(f'Frames processed: {i}')
        print(f'Total time: {end - global_start_time}s')

class Args:
    file: str | int
    verbose: bool
    name: str | None
    lin_buffer: int | None
    log_buffer: int | None
    kernel: int

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image feed obstruction removal.')

    parser.add_argument('-f', '--file', type=str, default=0, help='Path to video file or 0 for webcam feed.')

    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument('-n', '--name', type=str, default=None, help='Root name for buffers')
    parser.add_argument('-b', '--lin-buffer', type=int, default=None, help='Linear buffer size')
    parser.add_argument('-l', '--log-buffer', type=int, default=None, help='Log buffer size')
    parser.add_argument('-k', '--kernel', type=int, default=10, help='Kernel size for average pool convolutions')

    args: Args = parser.parse_args() # type: ignore

    # Process arguments

    if args.lin_buffer is None and args.log_buffer is None:
        raise ValueError('Must specify either linear or log buffer size.')

    if args.file == '0':
        video_input = 0
    else:
        video_input = args.file

    ## Frame buffers
    frame_buffers = []
    if args.lin_buffer is not None:
        frame_buffers.append(pf.FrameBuffer(args.lin_buffer, args.kernel, is_log_buffer=False))
    if args.log_buffer is not None:
        frame_buffers.append(pf.FrameBuffer(args.log_buffer, args.kernel, is_log_buffer=True))

    # Frame buffer names
    frame_buffer_names = None
    if args.name is not None:
        if len(frame_buffers) > 1:
            frame_buffer_names = [f'{args.name} linear', f'{args.name} log']
        else:
            frame_buffer_names = [args.name]

    # video_input = "../videos/terry_tao_low_res.mp4"

    image_feed = ImageFeed(
        video_input=video_input,
        frame_buffers=frame_buffers,
        frame_buffer_names=frame_buffer_names,
        verbose = args.verbose
    )

    image_feed.run_capture_loop()