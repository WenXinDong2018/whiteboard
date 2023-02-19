import cv2
import process_frame as pf
import numpy as np
import enhanced_frame as ef

####### From image feed START
video_input = "../videos/terry_tao_low_res.mp4"

vid = cv2.VideoCapture(video_input)

while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    ####### From image feed END

    enhanced_frame = ef.enhanced_frame(frame)

    ####### From image feed START
    # Display the resulting frame
    # print(frame)
    cv2.imshow('frame', enhanced_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
####### From image feed END
