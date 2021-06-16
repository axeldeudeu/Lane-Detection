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
