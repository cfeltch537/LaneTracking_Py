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
            print 'Remove:' + str(theta_degrees)

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

    half_size = (np.shape(binary_image)[0])/2
    binary_image[0:half_size] = 0
    print 'Removed Horizon'
    return binary_image


# def ConvertProbHoughPointsToHoughPoints(points):
#     lines = np.zeros(points.shape[0])
#     if points is not None:
#         for coordSet in points:
#             x1 = coordSet[0]
#             y1 = coordSet[1]
#             x2 = coordSet[2]
#             y2 = coordSet[3]
#             # Y = MX + B
#             M = (y2-y1)/(x2-x1)
#             # Find B, B = Y - MX
#             B = y1 - M*x1
#             # Find intercepts x0, and y0
#             # y0 = B
#             y0 = B
#             # x0 = -B/M
#             distance = x
#             lines = np.append([angle, distance])
#
#     return lines