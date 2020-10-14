import os
import cv2
import toml
import numpy as np
import fnmatch
import tqdm
from time import time


def undistort_images(config, images_path):
    intrinsics = toml.load(config)
    mtx = np.array(intrinsics['camera_mat'])
    dist = np.array(intrinsics['dist_coeff'])
    resolution = tuple([intrinsics['height'], intrinsics['width']])
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, resolution, 2, resolution)

    images = []
    for file in os.listdir(images_path):
        if fnmatch.fnmatch(file, '*.jpeg'):
            images.append(file)

    path_to_save_undistorted_images = os.path.join(images_path, 'undistorted')

    if not os.path.exists(path_to_save_undistorted_images):
        os.mkdir(path_to_save_undistorted_images)

    for image in tqdm.tqdm(images):
        img = cv2.imread(os.path.join(images_path, image))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        cv2.imwrite(os.path.join(path_to_save_undistorted_images, image), dst)


if __name__ =='__main__':
    config = './config/config_intrinsic.toml'
    image_path = './multimedia'
    undistort_images(config=config,images_path=image_path)
