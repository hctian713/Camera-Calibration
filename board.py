import cv2
import sys
import numpy as np
x_nums = 9
y_nums = 6
image = np.ones([(y_nums+2)*120, (x_nums+2)*120, 3], np.uint8) * 255

square_pixel = 120  # 1080/9 = 120 pixels
x0 = square_pixel
y0 = square_pixel


def DrawSquare():
    flag = -1
    for i in range(y_nums):
        flag = 0 - flag
        for j in range(x_nums):
            if flag > 0:
                color = [0, 0, 0]
            else:
                color = [255, 255, 255]
            cv2.rectangle(image, (x0 + j * square_pixel, y0 + i * square_pixel),
            (x0 + j * square_pixel + square_pixel, y0 + i * square_pixel + square_pixel), color, -1)
            flag = 0 - flag
        flag = 0 - flag
    cv2.imwrite('D:/chess_map_{}x{}.bmp'.format(x_nums,y_nums), image)

if __name__ == '__main__':
    DrawSquare()
