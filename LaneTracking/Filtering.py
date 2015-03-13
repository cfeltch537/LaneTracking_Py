# FILTERING MODULE
import cv2
import numpy as np


def CannyFilter(img, thresh_low, thresh_high):
    # CANNY FILTER,
    # INPUT - GRAYSCALE IMAGE
    # OUTPUT - CANNY IMAGE

    # BLUR THEN CANNY
    blurred_image = cv2.blur(img, (3, 3))
    canny_img = cv2.Canny(blurred_image, thresh_low, thresh_high, 3)
    return canny_img


def HoughFilter(canny_img, threshold):
    # HOUGH FILTER
    # INPUT - CANNY IMAGE
    # OUTPUT - LINES

    if threshold is 0:
        threshold = 1

    # HOUGH FILTER, RETURNING LINES
    lines = cv2.HoughLines(
        canny_img,  # input array
        1.0,  # rho
        np.pi / 180.0,  # theta
        threshold,  # threshold
        0, 0)  # length divisor, # angular resolution divisor
    return lines


def ProbabalisticHoughFilter(canny_img, threshold):
    # PROBABALISTIC HOUGH FILTER
    # INPUT - CANNY IMAGE
    # OUTPUT - LINES

    # threshold = cv2.getTrackbarPos(NAME_HOUGH_THRESHOLD, WINDOW_HOUGH)
    if threshold is 0:
        threshold = 1

    # HOUGH FILTER, RETURNING LINES
    lines = cv2.HoughLinesP(
        canny_img,  # input array
        1.0,  # rho
        np.pi / 180.0,  # theta
        threshold,  # threshold
        5, 0)  # length divisor, # angular resolution divisor
    return lines

