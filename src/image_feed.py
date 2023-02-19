import cv2

import process_frame as pf
import time

video_input = "../videos/terry_tao_low_res.mp4"

vid = cv2.VideoCapture(video_input)

frame_buffer_lin = pf.FrameBuffer(6, 10, is_log_buffer=True)
# frame_buffer_lin = pf.FrameBuffer(3, 10, is_log_buffer=False)

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    if not ret:
        break

    start = time.time()

    # Display the resulting frame
    # print(frame)
    # processed_frame_log = frame_buffer_log.process_frame(frame)
    processed_frame_lin, commit_frame_lin = frame_buffer_lin.process_frame(frame)

    # commit_frame_log = frame_buffer_log.commit_frame()
    # commit_frame_lin = frame_buffer_lin.commit_frame()
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