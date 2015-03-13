#!/usr/bin/env python


import numpy as np
import cv2

# Canny Filter
cannyThresholdLow_init = 90
cannyThresholdHigh_init = 190
CANNY_THRESHOLD_MAX = 200
WINDOW_CANNY = "Canny Filter"
NAME_CANNY_UPPER = 'Canny U'
NAME_CANNY_LOWER = 'Canny L'

# Hough Filter
hough_threshold = 80
HOUGH_MAX_THRESHOLD = 200
NAME_HOUGH_THRESHOLD = 'Hough T'
WINDOW_HOUGH = "Hough Filter"

input_image = None
grayscaled_image = None

LANE_MODE = 1


def CannyFilter(img):
    # CANNY FILTER,
    # INPUT - GRAYSCALE IMAGE
    # OUTPUT - CANNY IMAGE

    # GET ALL TRACKBAR VALUES
    canny_threshold_low = cv2.getTrackbarPos(NAME_CANNY_LOWER, WINDOW_CANNY)
    canny_threshold_high = cv2.getTrackbarPos(NAME_CANNY_UPPER, WINDOW_CANNY)

    # BLUR THEN CANNY
    blurred_image = cv2.blur(img, (3, 3))
    canny_img = cv2.Canny(blurred_image, canny_threshold_low, canny_threshold_high, 3)
    cv2.imshow(WINDOW_CANNY, canny_img)
    return canny_img


def HoughFilter(canny_img):
    # HOUGH FILTER
    # INPUT - CANNY IMAGE
    # OUTPUT - LINES

    threshold = cv2.getTrackbarPos(NAME_HOUGH_THRESHOLD, WINDOW_HOUGH)
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

def ProbabalisticHoughFilter(canny_img):
    # PROBABALISTIC HOUGH FILTER
    # INPUT - CANNY IMAGE
    # OUTPUT - LINES

    threshold = cv2.getTrackbarPos(NAME_HOUGH_THRESHOLD, WINDOW_HOUGH)
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
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
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


def RemoveAboveHorizon(binary_image):

    half_size = (np.shape(binary_image)[0])/2
    binary_image[0:half_size] = 0
    print 'Removed Horizon'
    return binary_image


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
    if LANE_MODE is 0:
        HORIZON_ANGLE_TO_REMOVE = 10
    else:
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


def clusterHoughPoints(lines):
    X = np.random.randint(25, 50, (25, 2))
    Y = np.random.randint(60, 85, (25, 2))
    Z = np.vstack((X, Y))

    # convert to np.float32
    Z = np.float32(Z)
    Z = lines

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, 2, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now separate the data, Note the flatten()
    A = Z[label.ravel() == 0]
    B = Z[label.ravel() == 1]

    return A, B


def getVanishingPoint(lines):
    return vanishing_point


def updateImage():
    global grayscaled_image

    # CANNY FILTER
    canny_img = CannyFilter(grayscaled_image)

    # bottom_half_img = RemoveAboveHorizon(canny_img)
    # HOUGH FILTER
    lines = HoughFilter(canny_img)
    # SEPARATE STREETS
    if lines is not None:
        # GET THE LINES ASSOCIATED WITH YOUR LANE ONLY, BY ANGLE
        street_lines = SeparateStreets(lines)
        # CLUSTER INTO LEFT AND RIGHT LANE (HOPEFULLY)
        if np.shape(street_lines)[0] > 2:
            cluster1, cluster2 = clusterHoughPoints(street_lines)
        else:
            cluster1 = None
            cluster2 = None
        # PLOT HOUGH LINES AND STREETS
        rgb_image = np.copy(cv2.cvtColor(grayscaled_image, cv2.COLOR_GRAY2BGR))
        all_lines_image = DrawHoughLinesOnImage(lines[0], rgb_image, (0, 0, 255))
        lane_seperated_image = DrawHoughLinesOnImage(street_lines, rgb_image, (0, 0, 255))
        street_lines_image = DrawHoughLinesOnImage(cluster1, rgb_image, (255, 255, 0))
        street_lines_image = DrawHoughLinesOnImage(cluster2, street_lines_image, (255, 0, 255))

        # SHOW HOUGH LINE IMAGES
        cv2.imshow(WINDOW_HOUGH, np.vstack((all_lines_image, lane_seperated_image, street_lines_image)))
        smaller_images = np.vstack((cv2.resize(all_lines_image, (0, 0), fx=0.5, fy=0.5), cv2.resize(lane_seperated_image, (0, 0), fx=0.5, fy=0.5), cv2.resize(street_lines_image, (0,0), fx=0.5, fy=0.5)))
        # cv2.imshow(WINDOW_HOUGH, smaller_images)

    # PROBABALISTIC HOUGH FILTER
    lines_p = ProbabalisticHoughFilter(canny_img)
    # SEPARATE STREETS
    if lines_p is not None:
        # # GET THE LINES ASSOCIATED WITH YOUR LANE ONLY, BY ANGLE
        # street_lines = SeparateStreets(lines)
        # # CLUSTER INTO LEFT AND RIGHT LANE (HOPEFULLY)
        # if np.shape(street_lines)[0] > 2:
        #     cluster1, cluster2 = clusterHoughPoints(street_lines)
        # else:
        #     cluster1 = None
        #     cluster2 = None
        # PLOT HOUGH LINES AND STREETS
        rgb_image_p = np.copy(cv2.cvtColor(grayscaled_image, cv2.COLOR_GRAY2BGR))
        all_lines_image_p = DrawProbHoughLinesOnImage(lines_p[0], rgb_image_p, (0, 0, 255))
        # lane_seperated_image_p = DrawHoughLinesOnImage(street_lines_p, rgb_image_p, (0, 0, 255))
        # street_lines_image_p = DrawHoughLinesOnImage(cluster1, rgb_image_p, (255, 255, 0))
        # street_lines_image_p = DrawHoughLinesOnImage(cluster2, street_lines_image_p, (255, 0, 255))
        # SHOW HOUGH LINE IMAGES
        cv2.imshow("WINDOW_HOUGH_PROB", all_lines_image_p)
        # cv2.imshow(WINDOW_HOUGH, np.vstack((all_lines_image_p, lane_seperated_image_p, street_lines_image_p)))
        # smaller_images = np.vstack((cv2.resize(all_lines_image, (0,0), fx=0.5, fy=0.5), cv2.resize(lane_seperated_image, (0,0), fx=0.5, fy=0.5), cv2.resize(street_lines_image, (0,0), fx=0.5, fy=0.5)))
        # cv2.imshow(WINDOW_HOUGH, smaller_images)


# define a trackbar callback
def onTrackbar(x):
    updateImage()


if __name__ == '__main__':
    # GET PHOTO
    filename = 'Street.jpg'
    input_image = cv2.imread(filename, 1)

    if input_image is None:
        print 'Cannot read input_image file: ' + filename

    # grayscaled_image.create(input_i mage.size(), input_image.type())
    grayscaled_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Create a WINDOW_CANNY
    cv2.namedWindow(WINDOW_CANNY, 1)
    cv2.namedWindow(WINDOW_HOUGH, 1)

    # Create Trackbars
    # Canny
    cv2.createTrackbar(NAME_CANNY_UPPER, WINDOW_CANNY, cannyThresholdHigh_init, CANNY_THRESHOLD_MAX, onTrackbar)
    cv2.createTrackbar(NAME_CANNY_LOWER, WINDOW_CANNY, cannyThresholdLow_init, CANNY_THRESHOLD_MAX, onTrackbar)
    # Houghq
    cv2.createTrackbar(NAME_HOUGH_THRESHOLD, WINDOW_HOUGH, hough_threshold, HOUGH_MAX_THRESHOLD, onTrackbar)

    # Show the input_image, explicitely call trackbar
    onTrackbar(0)

    writer = cv2.VideoWriter('Name.avi', -1, 30, (1280, 720))

    cap = cv2.VideoCapture()
    cap.open('StreetVideo.mp4')
    while cap.isOpened():
        ret, frame = cap.read()

        grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        updateImage()

        # cv2.imshow('frame', gray)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()

    # Wait for a key stroke the same function arranges events processing
    cv2.waitKey(0)
