# charuco calibration for behavioral rig

The goal is to use charuco chessboard to auto + real-time calibrate the
top camera (at least).

### an example chessboard would be like:

![Image text](multimedia/charuco.png)

### an example of video tracking the corners would be like:

![Image text](multimedia/demo-marker.gif)

### Calibration testing picture
1.  find out corners from the video/pictures/real-time recording
2.  calculate camera matrix
3.  calculate distortion coefficients
4.  save the camera matrix+coefficient+fps+resolution into .toml
5.  undistort the testing picture

![Image text](multimedia/test_1.jpeg)

### result (succeeded. Able to see the black residual on the right + bottom)

![Image text](multimedia/undistorted/test_1.jpeg)

=======
![Image text](https://raw.githubusercontent.com/caraido/charuco/master/multimedia/charuco.png)

### an example of video tracking the corners would be like:

![Image text](https://raw.githubusercontent.com/caraido/charuco/master/multimedia/demo-marker.gif)


