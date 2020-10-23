import PySpin
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import toml

import video_calib_3 as vc3

system = PySpin.System.GetInstance()

cam_list = system.GetCameras() # iterable of all the spinnaker cameras
cam = cam_list.GetByIndex(0)

# get node map
nodemap_tldevice = cam.GetTLDeviceNodeMap()
device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber')).GetValue()

## calibration related
# calibration board creation/load (default board)
board = vc3.create_board()
# record corners and Ids
allCorners =[]
allIds = []
max_size = vc3.get_expected_corners(board)
decimator = 3

# Initialize the camera
cam.Init()
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous) #or single frame, multiframe...
cam.TriggerMode.SetValue(PySpin.TriggerMode_Off) # or off if you want to trigger from software...
# etc
cam.BeginAcquisition() # if trigger is off, immediately collects the specified number of images

# get image height and width
nodemap = cam.GetNodeMap()
height = PySpin.CIntegerPtr(nodemap.GetNode('Height')).GetValue()
width = PySpin.CIntegerPtr(nodemap.GetNode('Width')).GetValue()

# openCV set video saving path
vid_path = './multimedia/'
vid_name = 'cam_' + device_serial_number + '_test.MOV'
# save the video using H.265 format
fourcc = cv2.VideoWriter_fourcc(*'HEVC')
out = cv2.VideoWriter(vid_path+vid_name, fourcc,30.0,(width, height))

while True:
    # see if the image is avalaible
    try:
        im = cam.GetNextImage()

        # may be unnecessary
        # width = im.GetWidth()
        # height = im.GetHeight()
        if im.IsIncomplete():
            print("Image incomplete with image status %d ..." % im.GetImageStatus())
            break

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)

    # convert the image with (1) pixel format and assign (2) the color process
    im_converted = im.Convert(PySpin.PixelFormat_Mono8, PySpin.NO_COLOR_PROCESSING)
    # get image pixels (in 1D array)
    im_pixel = im_converted.GetData()
    # reshape the pixels in (height, width)
    im_reshape = im_pixel.reshape(height, width)

    ## opencv take over
    # key step: detect markers
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshWinSizeMin = 100
    params.adaptiveThreshWinSizeMax = 700
    params.adaptiveThreshWinSizeStep = 50
    params.adaptiveThreshConstant = 5

    # get corners and refine them in openCV
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        im_reshape, board.dictionary, parameters=params)
    detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
        cv2.aruco.refineDetectedMarkers(im_reshape, board, corners, ids,
                                        rejectedImgPoints, parameters=params)

    # interpolate corners and draw corners
    if len(detectedCorners) > 0:
        rest, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
            detectedCorners, detectedIds, im_reshape, board)
        if detectedCorners is not None and 2 <= len(detectedCorners) <= max_size and decimator % 3 == 0:
            allCorners.append(detectedCorners)
            allIds.append(detectedIds)
        cv2.aruco.drawDetectedMarkers(im_reshape, corners, ids, borderColor=225)

    # write frame into the video
    out.write(im_reshape)
    cv2.imshow('frame', im_reshape)
    if cv2.waitKey(1) & 0xFF == 27:
        print('camera closed with esc key')
        break
    decimator += 1

    im.Release()

out.release()
cv2.destroyAllWindows()

cam.EndAcquisition()
cam.DeInit()
del cam
cam_list.Clear()
del cam_list
system.ReleaseInstance()
del system


# write in intrinsic configuration
output_path = './config/config_intrinsic_' + device_serial_number + '.toml'

if os.path.exists(output_path):
    print('\n config file already exists.')
else:
    calib = vc3.quick_calibrate(allCorners,allIds,board,width,height)
    calib['camera_serial_number'] = device_serial_number
    with open(output_path, 'w') as f:
        toml.dump(calib, f)
