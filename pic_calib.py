import matplotlib.pyplot as plt
import cv2

frame = cv2.imread("./multimedia/file")
plt.figure()
plt.imshow(frame)
plt.show()