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
