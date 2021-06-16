import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import argparse
import os

def histogram(image):
    # determine the normalized histogram
    m, n = image.shape
    hist = [0.0] * 256
    for i in range(m):
        for j in range(n):
            #for every intensity add the count
            hist[image[i, j]] += 1
    return np.array(hist)/(m*n)

def cumulativeSum(hist):
    # calculate the cumulative sum
    return [sum(hist[:i+1]) for i in range(len(hist))]

def histogramEqualization(image):
    #calculate Histogram
    hist = histogram(image)
    #find the cdf function
    cdf = np.array(cumulativeSum(hist))
    #multiply cdf with 255
    transfer = np.uint8(255 * cdf)
    k, l = image.shape
    final = np.zeros_like(image)
    # construct the final histogram equalization image
    for i in range(0, k):
        for j in range(0, l):
            final[i, j] = transfer[image[i, j]]
    return final


def gamma_correction(img,gamma):
    gamma = 1/gamma
    lT =[]
    for i in np.arange(0,256).astype(np.uint8):
        lT.append(np.uint8(((i/255)**gamma)*255))
    lookup = np.array(lT)
    #Creating the lookup table to find values
    corrected = cv2.LUT(img,lookup)
    return corrected

