# DRAWING OPERATIONS
import numpy as np
import cv2

def DrawHoughLinesOnImage(lines, image, color):
    # DRAW LINES ON IMAGE FOR TESTING/VISUAL
    # hough_image = np.copy(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    hough_image = np.copy(image)
    if lines is not None:
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            if np.isnan(a) or np.isnan(b) or np.isnan(x0) or np.isnan(y0):
                print "Found a NaN"
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(hough_image, (x1, y1), (x2, y2), color, 1)
    return hough_image

def DrawProbHoughLinesOnImage(lines, image, color):
    # DRAW LINES ON IMAGE FOR TESTING/VISUAL
    # hough_image = np.copy(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    hough_image = np.copy(image)
    if lines is not None:
        for line in lines:
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            cv2.line(hough_image, (x1, y1), (x2, y2), color, 2)
    return hough_image