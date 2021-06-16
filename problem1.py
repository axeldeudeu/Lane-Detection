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


def main(args):

    video = cv2.VideoWriter('Night_Drive_Correction.avi',cv2.VideoWriter_fourcc(*'XVID'), 20,(1024,600))
    cap = cv2.VideoCapture(args['file'])
    method = args['method']

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1024,600))
        #split in b,g,r
        b,g,r= cv2.split(frame)

        if (method == 'histogram'):
            #compute histogram equalization for each channel
            b1 = histogramEqualization(b)
            g1 = histogramEqualization(g)
            r1 = histogramEqualization(r)
            #merge the channels
            final = cv2.merge((b1,g1,r1))
        elif (method == 'gamma'):
            final = gamma_correction(frame, 1.8)
        else:
            print('invalid method ; exit')
            return

        cv2.imshow('Final', final)
        video.write(final)
        if cv2.waitKey(25) & 0XFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", "--method", required=True, help="Input: histogram or gamma", type=str)
    parser.add_argument("-path", "--file", required=False, help="video path", default='Night Drive - 2689.mp4', type=str)
    args = vars(parser.parse_args())
    if (not os.path.exists(args['file'])):
        print('File does not exist. Re run with correct path or place file in current directory and run')
        exit()

    main(args)
