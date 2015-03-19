# DRAWING OPERATIONS
import numpy as np
import cv2


def DrawHoughLinesOnImage(lines, image, color):
    # DRAW LINES ON IMAGE FOR TESTING/VISUAL
    # hough_image = np.copy(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    if lines is None:
        return image

    hough_image = np.copy(image)
    # Handle 2 dimensions, or 3
    if np.ndim(lines) is 2:
        lines = lines
    elif np.ndim(lines) is 3:
        lines = lines[0]
    else:
        print 'Strange number of dimensions? in ConvertProbHoughPointsToHoughPoints()'
        return None

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

    if lines is None:
        return image

    hough_image = np.copy(image)
    if lines is not None:
        for line in lines[0]:
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            cv2.line(hough_image, (x1, y1), (x2, y2), color, 2)
    return hough_image


def DrawLaneOnImage(lane, image, color):
    # DRAW LINES ON IMAGE FOR TESTING/VISUAL

    if lane is None:
        print "DrawLaneOnImage lane is None"
        return image

    hough_image = np.copy(image)

    if lane is not None:
        rho = lane[0]
        theta = lane[1]
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
        cv2.line(hough_image, (x1, y1), (x2, y2), color, 3)
    return hough_image


def DrawCenterLine(image, x, color):

    if color is None:
        return image
    ret_img = np.copy(image)
    x1 = int(x)
    y1 = np.shape(image)[0]
    x2 = int(x)
    y2 = np.shape(image)[0] - 50
    cv2.line(ret_img, (x1, y1), (x2, y2), color, 3)
    return ret_img


def ScaleAndStackImages(image_array, scale_factor):
    stacked_images = None
    for image in image_array:
        scaled_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        if stacked_images is None:
            stacked_images = scaled_image
        else:
            stacked_images = np.vstack((stacked_images, scaled_image))
    return stacked_images