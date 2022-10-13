import cv2
import numpy as np


# 3D坐标系构建函数
def square3D(img, imgpts):
    # 转换成坐标形式两列(x,y)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # square3D
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (229, 240, 53), 10)
    img = cv2.drawContours(img, [imgpts[:4]], -1, (229, 240, 53), 10)
    img = cv2.drawContours(img, [imgpts[4:8]], -1, (229, 240, 53), 10)

    return img


def axes3D(img, imgpts, oript):
    # 转换成坐标形式两列(x,y)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # axes3D
    img = cv2.line(img, tuple(oript[0]), tuple(imgpts[0]), (255, 0, 0), 20)
    img = cv2.line(img, tuple(oript[0]), tuple(imgpts[1]), (0, 255, 0), 20)
    img = cv2.line(img, tuple(oript[0]), tuple(imgpts[2]), (0, 0, 255), 20)
    return img















