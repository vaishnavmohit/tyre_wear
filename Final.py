
# coding: utf-8

# In[1]:


import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from random import *
import math
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[3]:


#http://www.jayrambhia.com/blog/disparity-mpas

def getDisparity(gray_left, gray_right):

    print(gray_left.shape)
    c, r = gray_left.shape
    stereoMatcher = cv2.StereoSGBM_create()
    stereoMatcher.setMinDisparity(3)#4)
    stereoMatcher.setNumDisparities(256)
    stereoMatcher.setBlockSize(19)
    stereoMatcher.setSpeckleRange(12)
    stereoMatcher.setSpeckleWindowSize(7)

    disparity = stereoMatcher.compute(gray_left, gray_right)#.astype(np.uint8)
    disparity_visual = cv2.normalize(disparity,disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return disparity_visual


# In[94]:


# test stereo pneu

tyre1 = 't21_c_1'
tyre2 = 't21_c_2'

dst1 = cv2.imread('tyre/'+tyre1+'.jpg',0)
dst2 = cv2.imread('tyre/'+tyre2+'.jpg',0)
c,r = dst1.shape
#c = .8*c
#r = .8*r

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=1.50, tileGridSize=(10,10))
dst1 = clahe.apply(dst1)
dst2 = clahe.apply(dst2)


# selection of SIFT points
sift = cv2.xfeatures2d.SIFT_create()

###find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(dst1,None)
kp2, des2 = sift.detectAndCompute(dst2,None)

###FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

###ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


pts1 = np.array(pts1)
pts2 = np.array(pts2)

#Computation of the fundamental matrix
F,mask= cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS) #least-median of squares algorithm istead FM_LMEDS, FM_RANSAC


# Obtainment of the rectification matrix and use of the warpPerspective to transform them...
pts1 = pts1[:,:][mask.ravel()==1]
pts2 = pts2[:,:][mask.ravel()==1]

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

p1fNew = pts1.reshape((pts1.shape[0] * 2, 1))
p2fNew = pts2.reshape((pts2.shape[0] * 2, 1))

retBool, rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(p1fNew,p2fNew,F,(r,c))

dst11 = cv2.warpPerspective(dst1,rectmat1,(r,c))
dst22 = cv2.warpPerspective(dst2,rectmat2,(r,c))

cv2.imwrite('rectified/'+tyre1+'_rectified.jpg', dst11)
cv2.imwrite('rectified/'+tyre2+'_rectified.jpg', dst22)


imgLeft = cv2.imread('rectified/'+tyre1+'_rectified.jpg',0)
imgRight = cv2.imread('rectified/'+tyre2+'_rectified.jpg',0)
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=1.50, tileGridSize=(10,10))
imgLeft = clahe.apply(imgLeft)
imgRight = clahe.apply(imgRight)

disparity = getDisparity(imgLeft, imgRight)

disparity[disparity>150] = 255
#disparity[disparity<2] = 255


plt.figure()
plt.imshow(disparity)#plt.figure()
#plt.imshow(imgLeft,'gray')
#plt.figure()
#plt.imshow(imgRight,'gray')
cv2.imwrite('disparity/'+tyre1+'_disparity.jpg',disparity)

ci,ri = disparity.shape
img1 = disparity[disparity<255]+.0001
dep = (ri*1.2*100/2340)*np.reciprocal((img1))


#avg_depth = 
total = 0.0 
avg_depth = 0.0
arr = np.unique(dep[dep<8.33])
num_element = arr.shape[0]
for i in range(num_element-3):
    total = (dep==arr[i]).sum() + total
    avg_depth = (dep==arr[i]).sum()*(arr[num_element-1]-arr[i]) + avg_depth
    #print(i)
    #print((dep==i).sum())
print(avg_depth/total)


file = open(tyre1+tyre2+'.txt','w') 
file.write("Images used are %s\r %s\n" %(tyre1,tyre2)) 
file.write("Avarage depth is %f\r\n" %(avg_depth/total)) 
file.close() 

