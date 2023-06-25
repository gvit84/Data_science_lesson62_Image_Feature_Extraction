import cv2
import numpy as np

img = cv2.imread('cat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(gray, None)
img_with_keypoints = cv2.drawKeypoints(gray, keypoints, None)

cv2.imwrite('SIFT Keypoints.jpg', img_with_keypoints)
cv2.imshow('SIFT Keypoints.jpg', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()