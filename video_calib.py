import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


# create a charuco board with grid_num * grid_num size
grid_num = 9
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # default
board = cv2.aruco.CharucoBoard_create(grid_num, grid_num, .025, .0125, dictionary)
img = board.draw((200 * 3, 200 * 3))

# Dump the calibration board to a file
#cv2.imwrite('./multimedia/charuco.png', img)

# turn on camera
path = './multimedia/test.MOV'
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print('open camera failed')
else:
    print('open camera succeeded')

# record corners and Ids
allCorners = []
allIds = []
decimator = 0

#
while True:

    ret, frame = cap.read()
    if not ret:
        break
    # turn the frame into grey scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # key step: detect markers, res[0]
    res = cv2.aruco.detectMarkers(gray, dictionary)

    if len(res[0]) > 0:
        #
        res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 3 == 0:
            allCorners.append(res2[1])
            allIds.append(res2[2])

        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == 27:
        print('camera closed with esc key')
        break
    decimator += 1

cap.release()
cv2.destroyAllWindows()

imsize = gray.shape

# Calibration fails for lots of reasons. Release the video if it fails
try:
    cal = cv2.aruco.calibrateCameraCharuco(allCorners, allIds, board, imsize, None, None)
    print("calibration succeeded!")
except:
    print("calibration failed!")
    cap.release()


# new camera matrix
mtx = cal[1]
dist_coef = cal[2]
rvec = cal[3]
trec = cal[4]
img = cv2.imread('./multimedia/test.jpeg')
w,  h = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist_coef,(w,h),0,(w,h))

# undistort
#dst = cv2.undistort(img, mtx, dist_coef, None, newcameramtx)
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist_coef,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
