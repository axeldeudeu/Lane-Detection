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

