import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import glob
import argparse
import os

class LaneDetection:

    def __init__(self):
        self.best_left_fit = np.array([1,1,1])
        self.best_right_fit = np.array([1,1,1])

    # function to read the camera parameters
    def readCameraParameters(self, dataset):
        K = np.array([[9.037596e+02, 0.00000000e+00, 6.957519e+02], [0.00000000e+00, 9.019653e+02, 2.242509e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        D = np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]])

        if (dataset == '2'):
            K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
            D = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])
        return K, D

    # function to undistort the image
    def undistortImage(self, image, K, D):
        return cv2.undistort(image,K,D,None,K)

    # function to get the warped image
    def getWarpedImage(self, image, src, dst):
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, H, (300, 300))
        return warped, H

    # function to do HLS color thresholding
    def doColorThresholding(self, warped):
        #https://www.w3schools.com/colors/colors_hsl.asp

        hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS).astype(np.float)

        #Seperate yellow
        lower_yellow = np.array([20,90,55],dtype=np.uint8)
        upper_yellow = np.array([45,200,255],dtype=np.uint8)
        yellow_mask = cv2.inRange(hls,lower_yellow,upper_yellow)
        yellow_line = cv2.bitwise_and(hls, hls, mask=yellow_mask).astype(np.uint8)

        #Seperate White
        lower_white = np.array([0,200,0],dtype=np.uint8)
        upper_white = np.array([255,255,255],dtype = np.uint8)
        white_mask = cv2.inRange(hls,lower_white,upper_white)
        white_line = cv2.bitwise_and(hls, hls, mask=white_mask).astype(np.uint8)

        mask = cv2.bitwise_or(white_mask,yellow_mask)
        preprocessed_hls = cv2.bitwise_or(yellow_line, white_line)
        preprocessed_img = cv2.bitwise_and(warped, warped, mask=mask).astype(np.uint8)

        return mask

    # function to generate histogram
    def generateHistogram(self, mask):
        hist = np.sum(mask, axis=0)
        midpoint = hist.shape[0]//2
        left_lane_ix = int(np.argmax(hist[:midpoint]))
        right_lane_ix = np.argmax(hist[midpoint:]) + midpoint
        return hist, left_lane_ix, right_lane_ix

    # function to fit a polynomial
    def fitPolynomial(self, mask, left_lane_list, right_lane_list, ones_x, ones_y, left_lane_x, left_lane_y, right_lane_x, right_lane_y):
        margin = 5
        try:
            left_fit = np.polyfit(left_lane_y, left_lane_x, 2)
            right_fit = np.polyfit(right_lane_y, right_lane_x, 2)
            # check for good coefficients and save this if good configuration
            if(self.best_left_fit is not None and self.best_right_fit is not None):
                if (abs(left_fit[1]-self.best_left_fit[1]) > 1.5):
                    #print('Take the last well known left fit')
                    left_fit = self.best_left_fit
                if (abs(right_fit[1]-self.best_left_fit[1]) > 1.5):
                    #print('Take the last well known right fit')
                    right_fit = self.best_right_fit
        except:
            #print('Error in fitting polynomial ; think of some method to solve and fit this')
            left_fit, right_fit = self.best_left_fit, self.best_right_fit
            #print(left_fit, right_fit)

        self.best_left_fit, self.best_right_fit = left_fit, right_fit

        # polynomial equation
        point_y = np.linspace(0, mask.shape[0]-1, mask.shape[0])
        left_line_x = left_fit[0]*(point_y**2) + left_fit[1]*(point_y) + left_fit[2]
        right_line_x = right_fit[0]*(point_y**2) + right_fit[1]*(point_y) + right_fit[2]

        # center line equation
        center_line_x = (left_line_x+right_line_x)/2
        center_fit = np.polyfit(point_y, center_line_x, 1)
        slope_center = center_fit[1]

        out_img = np.dstack((mask, mask, mask))
        window_img = np.zeros_like(out_img)

        # color mofication of non-zero pixels
        out_img[ones_y[left_lane_list], ones_x[left_lane_list]] = [255, 0, 0]
        out_img[ones_y[right_lane_list], ones_x[right_lane_list]] = [0, 255, 0]

        # Stack individual lane points
        left_line_window1 = np.array([np.transpose(np.vstack([left_line_x-margin, point_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_line_x+margin, point_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_line_x-margin, point_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_line_x+margin, point_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0,0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0))
        result = cv2.addWeighted(out_img, 1, window_img, 1, 0)

        # Stack the lane points together
        pts_left = np.array([np.transpose(np.vstack([left_line_x, point_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line_x, point_y])))])
        pts = np.hstack((pts_left, pts_right))

        #draw center line
        pts_center = np.array([np.transpose(np.vstack([center_line_x, point_y]))])
        cv2.polylines(result, np.int32([pts_center]), isClosed=False, color=(102,2,10), thickness=2)

        # Fill the lane polynomial
        cv2.fillPoly(result, np.int_([pts]),(0,0,255))

        return result, point_y, left_fit, right_fit, slope_center

    # function to predict turn
    def predict_turn(self, center_line_slope):
        if (center_line_slope > 160.0):
            return 'Prediction: Right Turn'
        if (center_line_slope >= 130.0 and center_line_slope <= 160.0):
            return 'Prediction: Go straight'
        if (center_line_slope < 130.0):
            return 'Prediction: Left Turn'
