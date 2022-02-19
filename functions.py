# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 20:17:23 2022

@author: David
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join
from scipy import ndimage
import cv2 as cv

def bordersToFeatures(ref):
    route = f'{ref}/'
    Areas = {}
    for border, f in enumerate(listdir(route)):
        if isfile(join(route, f)):
            im = cv.imread(join(route, f))
            imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(imgray, 127, 255, 0)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            closed_contours = []
            open_contours = []
            specified = {}
            by= []
            byy = []
            byyy = []
            for j, i in enumerate(contours):
                if cv.contourArea(i) > cv.arcLength(i, True):
                    by.append(cv.contourArea(i))
                    
                    x,y,w,h = cv.boundingRect(i)
                    aspect_ratio = float(w)/h
                    byy.append(aspect_ratio)
                    byyy.append(cv.arcLength(i, True))
                    specified[f'{cv.contourArea(i)}']=j
                    closed_contours.append(i)
                else:
                    byyy.append(0)
                    byy.append(0)
                    open_contours.append(i)
            cv.drawContours(im, contours, specified[str(np.max(by))], (0,255,0), 3)
            cv.imwrite(f'{ref}/contour_{border}.bmp', im)
            Areas[f] = [np.max(by), byy[specified[str(np.max(by))]], byyy[specified[str(np.max(by))]]]
    return Areas

def rgb2gray(rgb):
    gray = np.zeros(rgb.shape[:2], dtype = 'uint8')
    content = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    gray[:,:] = content

    return gray

def openImages(ref):
    route = f'{ref}/'
    imageNames = [join(route, f) for f in listdir(route) if isfile(join(route, f))]
    return np.array([mpimg.imread(name)/255.0 for name in imageNames])

def plotImages(images):
    for img in images:
        plt.figure()
        plt.imshow(img, 'Greys_r')
        
def saveImages(images, folderName, reference):
    mkdir(folderName)
    for i, img in enumerate(images):
        cv.imwrite(f'{folderName}/{reference}_{i}.bmp', img)

#Code from Sofiane Sahir to extract edges: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

def gaussian_kernel(size = 7, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def applyConvolve(images, kernel):
    return np.array([ndimage.convolve(image, kernel) for image in images])

def sobel_filters(imgs):
    newImgs = []
    thetas = []
    for img in imgs:
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        
        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)
        
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        
        newImgs.append(G)
        thetas.append(theta)
    return np.array(newImgs), np.array(thetas)
    #return (G, theta)

def non_max_suppression(imgs, Ds):
    Zs = []
    for number in range(len(imgs)):
        M, N = imgs[number].shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = Ds[number] * 180. / np.pi
        angle[angle < 0] += 180
    
        
        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255
                    
                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = imgs[number][i, j+1]
                        r = imgs[number][i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = imgs[number][i+1, j-1]
                        r = imgs[number][i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = imgs[number][i+1, j]
                        r = imgs[number][i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = imgs[number][i-1, j-1]
                        r = imgs[number][i+1, j+1]
    
                    if (imgs[number][i,j] >= q) and (imgs[number][i,j] >= r):
                        Z[i,j] = imgs[number][i,j]
                    else:
                        Z[i,j] = 0
                        
                except IndexError as e:
                    pass
        Zs.append(Z)
    return np.array(Zs)


def threshold(imgs, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    results = []
    weaks = []
    strongs = []
    for img in imgs:
        highThreshold = img.max() * highThresholdRatio;
        lowThreshold = highThreshold * lowThresholdRatio;
        
        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)
        
        weak = np.int32(25)
        strong = np.int32(255)
        
        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)
        
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
        
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        
        results.append(res)
        weaks.append(weak)
        strongs.append(strong)
    return (results, weaks, strongs)

def hysteresis(imgs, weaks, strong=255):
    finalImgs = []
    for number in range(len(imgs)):
        M, N = imgs[number].shape  
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (imgs[number][i,j] == weaks[number]):
                    try:
                        if ((imgs[number][i+1, j-1] == strong) or (imgs[number][i+1, j] == strong) or (imgs[number][i+1, j+1] == strong)
                            or (imgs[number][i, j-1] == strong) or (imgs[number][i, j+1] == strong)
                            or (imgs[number][i-1, j-1] == strong) or (imgs[number][i-1, j] == strong) or (imgs[number][i-1, j+1] == strong)):
                            imgs[number][i, j] = strong
                        else:
                            imgs[number][i, j] = 0
                    except IndexError as e:
                        pass
        finalImgs.append(imgs[number])
    return finalImgs