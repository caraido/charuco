#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import os
import cv2
import toml
import itertools
import numpy as np

from tqdm import trange
from time import time


def get_video_params(vid):
    cap = cv2.VideoCapture(vid)

    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['fps'] = cap.get(cv2.CAP_PROP_FPS)
    params['nframes'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return params


def get_expected_corners(board):
    board_size = board.getChessboardSize()
    return (board_size[0]-1)*(board_size[1]-1)


def get_corners_aruco_live(vid, board, skip=20):
    max_size = get_expected_corners(board)

    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print('open camera failed')
    else:
        print('open camera succeeded')

    # record corners and Ids
    allCorners = []
    allIds = []
    decimator = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # turn the frame into grey scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # key step: detect markers
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        params.adaptiveThreshWinSizeMin = 100
        params.adaptiveThreshWinSizeMax = 700
        params.adaptiveThreshWinSizeStep = 50
        params.adaptiveThreshConstant = 5

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, board.dictionary, parameters=params)
        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
            cv2.aruco.refineDetectedMarkers(gray, board, corners, ids,
                                            rejectedImgPoints, parameters=params)

        if len(detectedCorners) > 0:
            rest, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
                detectedCorners, detectedIds, gray, board)
            if detectedCorners is not None and 2 <= len(detectedCorners) <= max_size and decimator % 3 == 0:
                allCorners.append(detectedCorners)
                allIds.append(detectedIds)
            cv2.aruco.drawDetectedMarkers(gray, corners, ids, borderColor=225)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == 27:
            print('camera closed with esc key')
            break
        decimator += 1

    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['fps'] = cap.get(cv2.CAP_PROP_FPS)
    params['nframes'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    cv2.destroyAllWindows()

    return params, allCorners, allIds


def get_corners_aruco(vid, board, skip=20):

    cap = cv2.VideoCapture(vid)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    allCorners = []
    allIds = []

    go = int(skip / 2)

    max_size = get_expected_corners(board)

    for framenum in trange(length, ncols=70):
        ret, frame = cap.read()
        if not ret:
            break

        if framenum % skip != 0 and go <= 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        params.adaptiveThreshWinSizeMin = 100
        params.adaptiveThreshWinSizeMax = 700
        params.adaptiveThreshWinSizeStep = 50
        params.adaptiveThreshConstant = 5

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, board.dictionary, parameters=params)

        if corners is None or len(corners) <= 2:
            go = max(0, go - 1)
            continue

        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
            cv2.aruco.refineDetectedMarkers(gray, board, corners, ids,
                                            rejectedImgPoints, parameters=params)

        if len(detectedCorners) > 0:
            ret, detectedCorners, detectedIds = cv2.aruco.interpolateCornersCharuco(
                detectedCorners, detectedIds, gray, board)

        if detectedCorners is not None and \
                2 <= len(detectedCorners) <= max_size:
            allCorners.append(detectedCorners)
            allIds.append(detectedIds)
            go = int(skip / 2)

        go = max(0, go - 1)

    cap.release()

    return allCorners, allIds


def trim_corners(allCorners, allIds, maxBoards=85):
    '''
    only take "maxBoard" number of optimal allCorners
    '''
    counts = np.array([len(cs) for cs in allCorners])
    # detected more 6 corners
    sufficient_corners = np.greater_equal(counts, 6)
    sort = -counts + np.random.random(size=counts.shape) / 10
    subs = np.argsort(sort)[:maxBoards]
    allCorners = [allCorners[ix] for ix in subs if sufficient_corners[ix]]
    allIds = [allIds[ix] for ix in subs if sufficient_corners[ix]]
    return allCorners, allIds


def reformat_corners(allCorners, allIds):
    markerCounter = np.array([len(cs) for cs in allCorners])
    allCornersConcat = itertools.chain.from_iterable(allCorners)
    allIdsConcat = itertools.chain.from_iterable(allIds)

    allCornersConcat = np.array(list(allCornersConcat))
    allIdsConcat = np.array(list(allIdsConcat))

    return allCornersConcat, allIdsConcat, markerCounter


def calibrate_charuco(allCorners, allIds, board, video_params):
    print("\ncalibrating...")
    tstart = time()

    cameraMat = np.eye(3)
    distCoeffs = np.zeros(8)
    dim = (video_params['width'], video_params['height'])
    calib_flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3 + \
                  cv2.CALIB_FIX_PRINCIPAL_POINT
    calib_flags2 = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND +cv2.fisheye.CALIB_FIX_SKEW
    calib_flags3 = cv2.CALIB_RATIONAL_MODEL

    error, cameraMat, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        allCorners, allIds, board,
        dim, cameraMat, distCoeffs,
        flags=calib_flags3)

    tend = time()
    tdiff = tend - tstart
    print("\ncalibration took {} minutes and {:.1f} seconds".format(
        int(tdiff / 60), tdiff - int(tdiff / 60) * 60))

    out = dict()
    out['error'] = error
    out['camera_mat'] = cameraMat.tolist()
    out['dist_coeff'] = distCoeffs.tolist()
    out['width'] = video_params['width']
    out['height'] = video_params['height']
    out['fps'] = video_params['fps']

    return out


def calibrate_camera_aruco(vid, board):
    if vid == 0:
        video_params, someCorners, someIds = get_corners_aruco_live(vid, board)
    else:
        video_params = get_video_params(vid)
        someCorners, someIds = get_corners_aruco(vid, board)

    allCorners = []
    allIds = []

    allCorners.extend(someCorners)
    allIds.extend(someIds)

    allCorners, allIds = trim_corners(allCorners, allIds, maxBoards=100)
    allCornersConcat, allIdsConcat, markerCounter = reformat_corners(allCorners, allIds)

    expected_markers = get_expected_corners(board)

    print("\nfound {} markers, {} boards, {} complete boards".format(
        len(allCornersConcat), len(markerCounter),
        np.sum(markerCounter == expected_markers)))

    calib_params = calibrate_charuco(allCorners, allIds, board, video_params)
    return calib_params


# entrance
def calibrate_intrinsic(vid_path,board):
    output_path = './config/config_intrinsic.toml'

    if os.path.exists(output_path):
        print('\n config file already exists.')
    else:
        calib = calibrate_camera_aruco(vid_path, board)
        with open(output_path, 'w') as f:
            toml.dump(calib, f)

if __name__ == '__main__':
    # create a charuco board with grid_num * grid_num size
    grid_num = 9
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # default
    board = cv2.aruco.CharucoBoard_create(grid_num, grid_num, .025, .0125, dictionary)
    #img = board.draw((200 * 3, 200 * 3))

    path = './multimedia/test.MOV'
    calibrate_intrinsic(0, board)