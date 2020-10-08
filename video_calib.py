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
cv2.imwrite('charuco.png', img)


# turn on camera
cap = cv2.VideoCapture(0)
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray, dictionary)

    if len(res[0]) > 0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 3 == 0:
            allCorners.append(res2[1])
            allIds.append(res2[2])

        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == 27:
        print('camera closed')
        break
    decimator += 1

imsize = gray.shape

# Calibration fails for lots of reasons. Release the video if we do
try:
    cal = cv2.aruco.calibrateCameraCharuco(allCorners, allIds, board, imsize, None, None)
except:
    cap.release()

cap.release()
cv2.destroyAllWindows()

