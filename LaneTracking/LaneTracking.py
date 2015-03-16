#!/usr/bin/env python


import numpy as np
import cv2
import Filtering
import Operations
import DrawingOp as Draw

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


def updateImage():
    global grayscaled_image

    # CANNY FILTER
    canny_thresh_low = cv2.getTrackbarPos(NAME_CANNY_LOWER, WINDOW_CANNY)
    canny_thresh_high = cv2.getTrackbarPos(NAME_CANNY_UPPER, WINDOW_CANNY)
    canny_img = Filtering.CannyFilter(grayscaled_image, canny_thresh_low, canny_thresh_high)
    canny_img = Operations.RemoveAboveHorizon(canny_img)
    cv2.imshow(WINDOW_CANNY, canny_img)

    # HOUGH FILTER
    hough_threshold = cv2.getTrackbarPos(NAME_HOUGH_THRESHOLD, WINDOW_HOUGH)

    lines = Filtering.HoughFilter(canny_img, hough_threshold)

    # SEPARATE STREETS
    if lines is not None:
        # GET THE LINES ASSOCIATED WITH YOUR LANE ONLY, BY ANGLE
        street_lines = Operations.SeparateStreets(lines)
        # CLUSTER INTO LEFT AND RIGHT LANE (HOPEFULLY)
        if np.shape(street_lines)[0] > 2:
            cluster1, cluster2 = clusterHoughPoints(street_lines)
        else:
            cluster1 = None
            cluster2 = None
        # PLOT HOUGH LINES AND STREETS
        rgb_image = np.copy(cv2.cvtColor(grayscaled_image, cv2.COLOR_GRAY2BGR))
        all_lines_image = Draw.DrawHoughLinesOnImage(lines[0], rgb_image, (0, 0, 255))
        lane_seperated_image = Draw.DrawHoughLinesOnImage(street_lines, rgb_image, (0, 0, 255))
        street_lines_image = Draw.DrawHoughLinesOnImage(cluster1, rgb_image, (255, 255, 0))
        street_lines_image = Draw.DrawHoughLinesOnImage(cluster2, street_lines_image, (255, 0, 255))

        # SHOW HOUGH LINE IMAGES
        # cv2.imshow(WINDOW_HOUGH, np.vstack((all_lines_image, lane_seperated_image, street_lines_image)))
        smaller_images = np.vstack((cv2.resize(all_lines_image, (0, 0), fx=0.5, fy=0.5), cv2.resize(lane_seperated_image, (0, 0), fx=0.5, fy=0.5), cv2.resize(street_lines_image, (0,0), fx=0.5, fy=0.5)))
        cv2.imshow(WINDOW_HOUGH, smaller_images)

    # PROBABALISTIC HOUGH FILTER
    lines_p = Filtering.ProbabalisticHoughFilter(canny_img, hough_threshold)
    # lines_p = Filtering.ProbabalisticHoughFilter(canny_img, hough_threshold)
    lines_converted_p = Operations.ConvertProbHoughPointsToHoughPoints(lines_p)
    # SEPARATE STREETS
    if lines_converted_p is not None:
        # # GET THE LINES ASSOCIATED WITH YOUR LANE ONLY, BY ANGLE
        street_lines = Operations.SeparateStreets(lines)
        # # CLUSTER INTO LEFT AND RIGHT LANE (HOPEFULLY)
        # if np.shape(street_lines)[0] > 2:
        #     cluster1, cluster2 = clusterHoughPoints(street_lines)
        # else:
        #     cluster1 = None
        #     cluster2 = None
        # PLOT HOUGH LINES AND STREETS
        rgb_image_p = np.copy(cv2.cvtColor(grayscaled_image, cv2.COLOR_GRAY2BGR))
        all_lines_image_p = Draw.DrawProbHoughLinesOnImage(lines_p[0], rgb_image_p, (0, 0, 255))
        all_lines_image_p = Draw.DrawHoughLinesOnImage(street_lines, all_lines_image_p, (0, 255, 0))
        # lane_seperated_image_p = Draw.DrawHoughLinesOnImage(street_lines_p, rgb_image_p, (0, 0, 255))
        # street_lines_image_p = Draw.DrawHoughLinesOnImage(cluster1, rgb_image_p, (255, 255, 0))
        # street_lines_image_p = Draw.DrawHoughLinesOnImage(cluster2, street_lines_image_p, (255, 0, 255))
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
    filename = 'res/Street.jpg'
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

    # writer = cv2.VideoWriter('res/Highway.avi', -1, 30, (1280, 720))

    cap = cv2.VideoCapture()
    cap.open('res/Highway.avi')
    while cap.isOpened():
        ret, frame = cap.read()

        grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        updateImage()

        # cv2.imshow('frame', gray)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        if cv2.waitKey(20) & 0xFF == ord('1'):
            while True:
                if cv2.waitKey(20) & 0xFF == ord('2'):
                    break


    cap.release()

    # Wait for a key stroke the same function arranges events processing
    cv2.waitKey(0)
