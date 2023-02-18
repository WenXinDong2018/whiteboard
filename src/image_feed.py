import cv2
import process_frame as pf

video_input = "../videos/terry_tao_low_res.mp4"

vid = cv2.VideoCapture(0)

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    # print(frame)
    processed_frame = pf.process_frame(frame)
    cv2.imshow('frame', processed_frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()