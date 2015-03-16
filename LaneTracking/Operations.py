# OPERATIONS MODULE
# NOT QUITE FILTERS

import numpy as np


def SeparateStreets(lines):
    # Separate Street Lines - Remove lines associated with horizon
    # INPUT - CANNY IMAGE
    # OUTPUT - LINES

    new_lines = np.copy(lines[0])
    indices_for_removal = np.array([])
    for i in range(np.shape(new_lines)[0]):
        rho_theta = new_lines[i]
        theta_degrees = rho_theta[1] * 180 / np.pi

        # if isCloseToHorizontal(theta_degrees) or isCloseToVertical(theta_degrees):
        if isMarkingsOfCurrentLane(theta_degrees):
            indices_for_removal = np.append(indices_for_removal, [i])
            # print 'Removed:' + str(theta_degrees)

    new_lines = np.delete(new_lines, indices_for_removal, axis=0)
    return new_lines


def isMarkingsOfCurrentLane(theta_degrees):
    # Returns true if 'theta_degrees' is within 90 +- HORIZON_ANGLE_TO_REMOVE
    ANGLE_TO_REMOVE = 25
    if np.greater_equal(theta_degrees, 90 - ANGLE_TO_REMOVE) \
            and np.less_equal(theta_degrees, 90 + ANGLE_TO_REMOVE):
        return True
    else:
        return False


def isCloseToHorizontal(theta_degrees):
    # Returns true if 'theta_degrees' is within 90 +- HORIZON_ANGLE_TO_REMOVE
    HORIZON_ANGLE_TO_REMOVE = 10

    if np.greater_equal(theta_degrees, 90 - HORIZON_ANGLE_TO_REMOVE) \
            and np.less_equal(theta_degrees, 90 + HORIZON_ANGLE_TO_REMOVE):
        return True
    else:
        return False


def isCloseToVertical(theta_degrees):
    # Returns true if 'theta_degrees' is within 90 +- HORIZON_ANGLE_TO_REMOVE
    VERTICAL_ANGLE_TO_REMOVE = 1
    if np.greater_equal(theta_degrees, 180 - VERTICAL_ANGLE_TO_REMOVE) \
            or np.less_equal(theta_degrees, 0 + VERTICAL_ANGLE_TO_REMOVE):
        return True
    else:
        return False


def RemoveAboveHorizon(binary_image):
    half_size = (np.shape(binary_image)[0]) / 2
    binary_image[0:half_size] = 0
    print 'Removed Horizon'
    return binary_image


def ConvertProbHoughPointsToHoughPoints(points):
    lines = None
    if points is not None:
        if np.ndim(points) is 2:
            points = points
        else:
            points = points[0]
        for coordSet in points:
            coordSet_float = coordSet.astype(np.float32)
            x1 = coordSet_float[0]
            y1 = coordSet_float[1]
            x2 = coordSet_float[2]
            y2 = coordSet_float[3]
            # Y = MX + B
            # Find B, B = Y - MX
            # Find intercepts x_int, and y_int
            # y_int = B, # x_int = -B/M
            M = float((y2 - y1) / (x2 - x1))
            B = float(y1 - M * x1)
            if M == float('+inf') or M == float('-inf'):  # VERTICAL LINE
                x_int = x1  # or x2
                theta = float(0.0)
                rho = x_int * np.cos(theta)
            elif M == float(0):  # HORIZONTAL LINE
                y_int = y1  # or y2
                theta = float(np.pi/2)
                rho = y_int * np.sin(theta)
            else:
                y_int = B
                x_int = -B / M
                x0 = B / (1/M + M)
                # y0 =
                theta = (90 * np.pi/180) - np.arctan2(y_int, x_int)
                rho = - x0 / np.cos(theta)

            # theta_degrees = theta * (180 / np.pi)
            if np.isnan(theta) or np.isnan(rho):
                print "Look NaN!"

            hough_point = np.array([[rho, theta]])

            # print "Original " + str(points) + " -> " + str(hough_point)

            if lines is None:
                lines = np.copy(hough_point)
            else:
                lines = np.concatenate((lines, hough_point))

    if lines is not None:
        return lines.astype(np.float32)
    else:
        return lines