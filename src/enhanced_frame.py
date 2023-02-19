import cv2
import numpy as np

def enhanced_frame(frame):
    rgb_planes = cv2.split(frame)
    for plane in rgb_planes:

        # Increases the object area used to accentuate features
        dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))

        # This operation processes the edges while removing the noise.
        bg_img = cv2.medianBlur(dilated_img, 21)

        # Absdiff calculates the per-element absolute difference between two arrays
        diff_img = 255 - cv2.absdiff(plane, bg_img)

        # Normalize the image
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # Threshold the image
        th = cv2.adaptiveThreshold(norm_img,
                                   240,  # maximum value assigned to pixel values exceeding the threshold
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # gaussian weighted sum of neighborhood
                                   cv2.THRESH_BINARY,  # thresholding type
                                   3,  # block size (5x5 window)
                                   7)  # constant
        # Invert
        bh = np.invert(norm_img)

    return bh


# Path: Otsu Thresholding
# # Convert to grayscale
# gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
#
# # Apply median filter
# median = cv2.medianBlur(gray, 5)
#
# # Apply adaptive threshold
# thresh = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#
# # Apply dilation and erosion to remove some noise
# kernel = np.ones((1, 1), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
#
# # Apply Otsu threshold
# ret, otsu = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
# return otsu


## ONE ATTEMPT ##
# ret, foreground = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
# ret, shadow = cv2.threshold(frame, 200, 255, cv2.THRESH_TOZERO_INV)
# # stack images vertically
# res = np.concatenate((frame, foreground, shadow), axis=0)
# return res
