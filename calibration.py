import cv2
import numpy as np
import glob
import draw

"""相机标定"""
# 设置寻找亚像素角点的最佳迭代终止条件，最大循环次数30和最大误差容县0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
# 获取标定板角点的位置
world = np.zeros((5 * 8, 3), np.float32)
# 将世界坐标系建在标定板上，所有点的Z坐标全部是0
world[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)
# 存储世界坐标和图像坐标的角点
world_points = []
image_points = []

# 读入棋盘格影像
images = glob.glob('chess9x6/IMG*.jpg')  # 输入图像路径
calibrated_images = []
i = 1
for image in images:
    img = cv2.imread(image)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 把BGR图像转化为灰度图像
    size = gray_image.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray_image, (8, 5), None)  # 角点检测
    print(ret)
    if ret:  # 如果检测成功
        world_points.append(world)
        # 寻找亚像素点
        corners_subpixel = cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
        if [corners_subpixel]:
            image_points.append(corners_subpixel)
        else:
            image_points.append(corners)
        cv2.drawChessboardCorners(img, (8, 5), corners, ret)  # 画出成功检测的角点
        calibrated_images.append(img)
        cv2.imwrite('calibrated_imgs/imgcal{}.jpg'.format(i), img)
        i += 1

print(len(image_points))  # 图像的数量

# 相机标定
ret, camera_matrix, distortion_coefficient, r_vectors, t_vectors = cv2.calibrateCamera \
    (world_points, image_points, size, None, None)
# 将旋转向量转化为旋转矩阵
r_matrixs = []
for r_vec in r_vectors:
    r_matrixs.append(cv2.Rodrigues(r_vec.reshape(1, 3)[0])[0])  # 罗德里格斯公式转换

# 输出相机的内参数矩阵、畸变系数、旋转矩阵和平移向量
print("camera matrix:\n", camera_matrix)  # 内参数矩阵
print("distortion coefficient:\n", distortion_coefficient)  # 畸变系数(k_1,k_2,p_1,p_2,k_3)
print("rotation vectors:\n", r_vectors)  # 旋转向量  # 外参数
print("rotation matrixs:\n", r_matrixs)  # 旋转矩阵
print("translation vectors:\n", t_vectors)  # 平移向量  # 外参数

with open('dst.txt', 'w') as f:
    f.write("image numbers\n" + str(len(image_points)) + "\n")
    f.write("intrinsic matrix\n" + str(camera_matrix) + "\n")
    f.write("translation vectors:\n" + str(t_vectors) + "\n")
    f.write("rotation vectors:\n" + str(r_vectors) + "\n")
    f.write("rotation matrixs:\n" + str(r_matrixs) + "\n")
    f.write("distortion coefficient:\n" + str(distortion_coefficient))

"""矫正图影像；计算精度；绘制坐标系和立方体"""
# 读入标定好的影像
imgcals = glob.glob('calibrated_imgs/*.jpg')  # 输入图像路径
# 3Dsquare 1-8,axes3D，9-11
wdpts = np.float32([[0, 0, 0], [0, 2, 0], [2, 2, 0], [2, 0, 0], [0, 0, -2], [0, 2, -2], [2, 2, -2], [2, 0, -2],
                    [4, 0, 0], [0, 4, 0], [0, 0, -4]])

# 对每一幅标定好的影像进行处理
for i in range(len(image_points)):
    i += 1
    # 获取优化后的相机内参
    img = cv2.imread('calibrated_imgs/imgcal{}.jpg'.format(i))
    h, w = img.shape[:2]
    print(img.shape[:2])
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficient, (w, h), 1, (w, h))
    print("img{}-roi:".format(i) + str(roi))
    # 图像校正
    imgudt = cv2.undistort(img, camera_matrix, distortion_coefficient, None, new_camera_matrix)
    cv2.imwrite('undistort_imgs/imgudt{}.jpg'.format(i), imgudt)
    # 绘制
    # 用PnP算法获取旋转矩阵和平移向量
    # _, r_vector, t_vector, inliers = cv2.solvePnPRansac(world, corners_subpixel, camera_matrix,distortion_coefficient)
    # 重投影
    imgpts, jac = cv2.projectPoints(wdpts, r_vectors[i - 1], t_vectors[i - 1], camera_matrix,
                                    distortion_coefficient)
    img1 = draw.axes3D(img, imgpts[-3:], imgpts[0])
    outcome_image = draw.square3D(img1, imgpts[:8])
    cv2.imwrite('dst_imgs/haochen_tian_{}.jpg'.format(i), outcome_image)

# 重投影误差
total_error = 0
for j in range(len(world_points)):
    image_points2, _ = cv2.projectPoints(world_points[j], r_vectors[j], t_vectors[j], camera_matrix,
                                         distortion_coefficient)
    error = cv2.norm(image_points[j], image_points2, cv2.NORM_L2) / len(image_points2)
    total_error += error
    mean_error = total_error / len(world_points)
print("Reprojection error: " + str(mean_error))
