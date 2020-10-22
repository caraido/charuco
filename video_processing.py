import cv2
import PySpin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class CharucoBoard:
    def __init__(self, grid_num, marker_size):
        self.grid_num = grid_num
        self.marker_size = marker_size
        self.default_dictionary = cv2.aruco.DICT_4X4_50 # default
        self.seed = 0
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.default_dictionary)

    @property
    def board(self):
        this_board = cv2.aruco.CharucoBoard_create(self.grid_num,
                                             self.grid_num,
                                             1,
                                             self.marker_size,
                                             self.dictionary)
        return this_board

    @property
    def marker_size(self):
        return self._marker_size

    @marker_size.setter
    def marker_size(self,value):
        if value <=0 or value >=1:
            raise ValueError("this value can only be set between 0 ~ 1!")
        else:
            self._marker_size = value

    def save_board(self, img_size=1000):
        if not self.default_dictionary:
            file_name = 'charuco_board_grid#_' + str(self.grid_num) + '_marker_size_'+ str(self.marker_size) + 'default.png'
            img = self.board.draw((img_size, img_size))
            cv2.imwrite('./multimedia/board/'+file_name, img)
        else:
            pass

    def print_board(self):
        img = self.board.draw((1000,1000))
        plt.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")
        plt.axis("off")
        plt.show()


class Camera:
    def __init__(self):
        pass

    def get_all_cameras(self):
        pass

    def start_camera(self):
        pass

    def get_camera_property(self):
        pass

    def calibration(self):
        pass

    def acquire_image(self):
        pass

    def save_image(self):
        pass

    def save_video(self):
        pass

    def save_intrinsic(self):
        pass

    def save_extrinsic(self):
        pass
