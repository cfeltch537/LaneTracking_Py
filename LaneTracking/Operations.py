# OPERATIONS MODULE
# NOT QUITE FILTERS

import numpy as np
import cv2


def SeparateStreets(lines):
    # Separate Street Lines - Remove lines associated with horizon
    # INPUT - CANNY IMAGE
    # OUTPUT - LINES

    if lines is None:
        return None

    # Handle 2 dimensions, or 3
    new_lines = None
    if np.ndim(lines) is 2:
        new_lines = np.copy(lines)
    elif np.ndim(lines) is 3:
        new_lines = np.copy(lines[0])
    else:
        print 'Strange number of dimensions? in SeparateStreets()'
        return new_lines

    indices_for_removal = np.array([])
    for i in range(np.shape(new_lines)[0]):
        rho_theta = new_lines[i]
        theta_degrees = rho_theta[1] * 180 / np.pi

        # if isCloseToHorizontal(theta_degrees) or isCloseToVertical(theta_degrees):
        if isMarkingsOfCurrentLane(theta_degrees):  # and not isCloseToVertical(theta_degrees)
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
    VERTICAL_ANGLE_TO_REMOVE = 2
    if np.greater_equal(theta_degrees, 180 - VERTICAL_ANGLE_TO_REMOVE) \
            or np.less_equal(theta_degrees, 0 + VERTICAL_ANGLE_TO_REMOVE):
        return True
    else:
        return False


def RemoveAboveHorizon(binary_image, offset):

    half_height = (np.shape(binary_image)[0]) / 2
    binary_image[0:half_height - offset] = 0
    x1 = 0
    x2 = np.shape(binary_image)[1]
    y1 = half_height - offset
    y2 = half_height - offset
    points_of_line = np.array([x1, y1, x2, y2])

    return binary_image, points_of_line


def ConvertProbHoughPointsToHoughPoints(points):
    lines = None

    if points is None:
        return None
    else:
        # Handle 2 dimensions, or 3
        if np.ndim(points) is 2:
            points = points
        elif np.ndim(points) is 3:
            points = points[0]
        else:
            print 'Strange number of dimensions? in ConvertProbHoughPointsToHoughPoints()'
            return None

        # For each coordinate convert from Cartesian to Hough Space
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
                theta = (90 * np.pi/180) - np.arctan2(y_int, x_int)
                rho = - x0 / np.cos(theta)

            # theta_degrees = theta * (180 / np.pi)
            if np.isnan(theta) or np.isnan(rho):
                print "Look NaN!"

            hough_point = np.array([[rho, theta]])

            if lines is None:
                lines = np.copy(hough_point)
            else:
                lines = np.concatenate((lines, hough_point))

    if lines is not None:
        return lines.astype(np.float32)
    else:
        return lines


def ClusterHoughPoints(lines):

    if lines is None or np.shape(lines)[0] < 2:
        return None, None

    Z = lines

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now separate the data, Note the flatten()
    A = Z[label.ravel() == 0]
    B = Z[label.ravel() == 1]

    return A, B


def DetermineLanes(cluster1, cluster2, old_left_lane, old_right_lane):

    if cluster1 is None and cluster2 is None:
        # print "DetermineLanes: Both clusters are None"
        return old_left_lane, old_right_lane

    lane1 = GetLaneFromMedian(cluster1)
    lane2 = GetLaneFromMedian(cluster2)

    # Assume the right lane has a smaller angle than
    if areClusterAnglesTooClose(lane1, lane2):
        # The two clusters are very close, assume they are one cluster!
        combined_clusters = np.vstack((cluster1, cluster2))
        single_lane = GetLaneFromMedian(combined_clusters)

        diff_from_old_left = abs(single_lane[1] - old_left_lane[1])
        diff_from_old_right = abs(single_lane[1] - old_right_lane[1])

        if diff_from_old_left < diff_from_old_right:
            # Assume Cluster is the Left Lane
            return single_lane, old_right_lane
        else:
            # Assume Cluster is the Right Lane
            return old_left_lane, single_lane
    else:
        if lane1[1] < lane2[1]:
            # Assume lane1 is the right lane
            left_lane = lane1
            right_lane = lane2
        else:
            # Assume lane2 is the right lane
            right_lane = lane1
            left_lane = lane2

    # if lane1 is None:
    #     if abs(lane2[1] - old_left_lane[1]) < abs(lane2[1] - old_right_lane):
    #         # ASSUME LEFT LANE
    #         return lane2, old_right_lane
    #     else:
    #         return old_left_lane, lane2
    # if lane2 is None:
    #     if abs(lane1[1] - old_left_lane[1]) < abs(lane1[1] - old_right_lane):
    #         # ASSUME LEFT LANE
    #         return lane1, old_right_lane
    #     else:
    #         return old_left_lane, lane1
    #
    # if lane1[1] < lane2[1]:
    #     left_lane = lane1
    #     right_lane = lane2
    # else:
    #     right_lane = lane1
    #     left_lane = lane2

    return left_lane, right_lane


def areClusterAnglesTooClose(cluster_avg_1, cluster_avg_2):
    MIN_DISTANCE = (15 * np.pi/180)  # 10 Degrees
    return abs(cluster_avg_1[1] - cluster_avg_2[1]) < MIN_DISTANCE


def GetLaneFromMedian(cluster):

    if cluster is None:
        print "GetLaneFromMedian: cluster is empty"
        return None

    distances = cluster[:, 0]
    distances = np.sort(distances)
    angles = cluster[:, 1]
    angles = np.sort(angles)
    avg_angle = np.median(angles)
    avg_distance = np.median(distances)

    return np.array([avg_distance, avg_angle])


def GetCenterPointBetweenLanes(left_lane, right_lane, image):
    center_x = None

    x1 = GetInterceptX(left_lane, image)
    x2 = GetInterceptX(right_lane, image)

    x_center = abs(x1 - x2)/2 + np.min([x1, x2])

    return x_center, x1, x2


def GetInterceptX(lane, image):

    # Find x intercepts of each lane, then return center point
    theta = lane[1]
    rho = lane[0]
    phi = 90*np.pi/180 + theta
    M = np.sin(phi)/np.cos(phi)
    x0 = rho*np.cos(theta)
    y0 = rho*np.sin(theta)
    y1 = np.shape(image)[1]/2
    x1 = (M*x0-y0+y1)/M

    return x1