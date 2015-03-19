#!/usr/bin/env python


import numpy as np
import cv2
import Filtering
import Operations
import DrawingOp as Draw

# Ignoring Segments
horizon_offset_origin = 100
horizon_max = 200
NAME_HORIZON_SLIDER = "H Off"

# Canny Filter
cannyThresholdLow_init = 70
cannyThresholdHigh_init = 130
CANNY_THRESHOLD_MAX = 200
WINDOW_CANNY = "Canny Filter"
NAME_CANNY_UPPER = 'Canny U'
NAME_CANNY_LOWER = 'Canny L'

# Hough Filter
hough_threshold = 50
HOUGH_MAX_THRESHOLD = 100
NAME_HOUGH_THRESHOLD = 'Hough T'
WINDOW_HOUGH = "Hough Filter"

input_image = None
grayscaled_image = None

LANE_MODE = 1

left_outer = None
left_inner = None
right_inner = None
right_outer = None

def updateImage():
    global grayscaled_image

    # GET SLIDER BAR VALUES
    canny_thresh_low = cv2.getTrackbarPos(NAME_CANNY_LOWER, WINDOW_CANNY)
    canny_thresh_high = cv2.getTrackbarPos(NAME_CANNY_UPPER, WINDOW_CANNY)
    hough_thresh = cv2.getTrackbarPos(NAME_HOUGH_THRESHOLD, WINDOW_HOUGH)
    horizontal_thresh = cv2.getTrackbarPos(NAME_HORIZON_SLIDER, WINDOW_CANNY)

    # CANNY FILTER
    canny_img = Filtering.CannyFilter(grayscaled_image, canny_thresh_low, canny_thresh_high)

    # REMOVE ABOVE HORIZON
    removed_horizon_img, horizontal_line = Operations.RemoveAboveHorizon(canny_img, horizontal_thresh - horizon_offset_origin)
    removed_horizon_img_w_lines = np.copy(cv2.cvtColor(removed_horizon_img, cv2.COLOR_GRAY2BGR))
    cv2.line(removed_horizon_img_w_lines, (horizontal_line[0], horizontal_line[1]), (horizontal_line[2], horizontal_line[3]), (0,0,255), 2)
    cv2.imshow(WINDOW_CANNY, removed_horizon_img_w_lines)

    # HOUGH FILTER
    lines = Filtering.HoughFilter(removed_horizon_img, hough_thresh)
    # all_lines_image = Draw.DrawHoughLinesOnImage(lines, rgb_image, (0, 0, 255))

    # SEPARATE STREETS, GET THE LINES ASSOCIATED WITH YOUR LANE ONLY, BY ANGLE
    street_lines = Operations.SeparateStreets(lines)
    # lane_separated_image = Draw.DrawHoughLinesOnImage(street_lines, rgb_image, (0, 0, 255))

    # CLUSTER INTO LEFT AND RIGHT LANE (HOPEFULLY)
    cluster1, cluster2 = Operations.ClusterHoughPoints(street_lines)
    cluster11, cluster12 = Operations.ClusterHoughPoints(cluster1)
    # lane11 = Operations.GetLaneFromStdDeviation(cluster11)
    # lane12 = Operations.GetLaneFromStdDeviation(cluster12)
    cluster21, cluster22 = Operations.ClusterHoughPoints(cluster2)
    # lane21 = Operations.GetLaneFromStdDeviation(cluster21)
    # lane22 = Operations.GetLaneFromStdDeviation(cluster22)

    # FIND AND DRAW LINES ASSOCIATED WITH STREETS
    rgb_image = np.copy(cv2.cvtColor(grayscaled_image, cv2.COLOR_GRAY2BGR))
    street_lines_image = Draw.DrawHoughLinesOnImage(cluster11, rgb_image, (255, 255, 0))
    street_lines_image = Draw.DrawHoughLinesOnImage(cluster12, street_lines_image, (255, 0, 255))
    street_lines_image = Draw.DrawHoughLinesOnImage(cluster21, street_lines_image, (255, 0, 0))
    street_lines_image = Draw.DrawHoughLinesOnImage(cluster22, street_lines_image, (0, 0, 255))

    # FIND AND DRAW LANES
    global left_outer, left_inner, right_inner, right_outer
    left_outer, left_inner, right_inner, right_outer = Operations.\
        DetermineLanes(cluster1, cluster2, left_outer, left_inner, right_inner, right_outer)
    lanes_image = np.copy(rgb_image)
    # lanes_image = Draw.DrawLaneOnImage(left_lane, lanes_image, (255, 0, 0))
    # lanes_image = Draw.DrawLaneOnImage(right_lane, lanes_image, (255, 0, 255))

    lanes_image = Draw.DrawLaneOnImage(left_outer, lanes_image, (255, 0, 0))
    lanes_image = Draw.DrawLaneOnImage(left_inner, lanes_image, (255, 255, 0))
    lanes_image = Draw.DrawLaneOnImage(right_outer, lanes_image, (0, 255, 0))
    lanes_image = Draw.DrawLaneOnImage(right_inner, lanes_image, (0, 0, 255))

    # DRAW CENTER LINE IMAGE FOR FRAME
    lanes_w_center_line = Draw.DrawCenterLine(lanes_image, np.shape(lanes_image)[1]/2, (0, 255, 0))

    # GET AND DRAW CENTER LINE FOR LANE X INTERCEPTS
    center_point_x, x_left, x_right = Operations.GetCenterPointBetweenLanes(left_inner, right_inner, lanes_w_center_line)
    lanes_w_center_line = Draw.DrawCenterLine(lanes_w_center_line, center_point_x, (255, 255, 0))

    # DISPLAY PERCENT FROM LEFT AND RIGHT LANE ON IMAGE
    image_width = np.shape(lanes_w_center_line)[1]
    image_height = np.shape(lanes_w_center_line)[0]
    percent_from_left = (image_width/2 - x_left) / (x_right - x_left)
    percent_from_right = (image_width/2 - x_right) / (x_right - x_left)
    left_text = 'Left Lane = ' + str(int(percent_from_left*100))
    right_text = 'Right Lane = ' + str(int(percent_from_right*100))
    cv2.putText(lanes_w_center_line, left_text, (0, image_height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), thickness=2)
    cv2.putText(lanes_w_center_line, right_text, (300, image_height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), thickness=2)

    # SHOW HOUGH LINE IMAGES
    images = np.array([street_lines_image, lanes_w_center_line])
    smaller_images = Draw.ScaleAndStackImages(images, 1.0)
    cv2.imshow(WINDOW_HOUGH, smaller_images)


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
    cv2.createTrackbar(NAME_HORIZON_SLIDER, WINDOW_CANNY, horizon_offset_origin, horizon_max, onTrackbar)
    # Hough
    cv2.createTrackbar(NAME_HOUGH_THRESHOLD, WINDOW_HOUGH, hough_threshold, HOUGH_MAX_THRESHOLD, onTrackbar)

    # Show the input_image, explicitly call trackbar
    onTrackbar(0)

    # writer = cv2.VideoWriter('res/Highway.avi', -1, 30, (1280, 720))

    cap = cv2.VideoCapture()
    cap.open('res/road.avi')
    while cap.isOpened():
        ret, frame = cap.read()

        grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        updateImage()

        # cv2.imshow('frame', gray)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

        if cv2.waitKey(5) & 0xFF == ord('1'):
            while True:
                if cv2.waitKey(20) & 0xFF == ord('2'):
                    break

    cap.release()

    # Wait for a key stroke the same function arranges events processing
    cv2.waitKey(0)
