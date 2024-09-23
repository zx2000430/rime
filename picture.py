import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


# 参数设置
nx = 9
ny = 6
file_paths = glob.glob("./camera_cal/calibration*.jpg")

# 绘制对比图
def plot_contrast_image(origin_img, converted_img, origin_img_title="origin_img", converted_img_title="converted_img",
                        converted_img_gray=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 20))
    ax1.set_title(origin_img_title)
    ax1.imshow(origin_img, cmap=None)
    ax2.set_title(converted_img_title)
    if converted_img_gray:
        ax2.imshow(converted_img, cmap="gray")
    else:
        ax2.imshow(converted_img, cmap=None)
    plt.show()

# 相机校正：外参，内参，畸变系数
def cal_calibrate_params(file_paths):
    # 存储角点数据的坐标90
    object_points = []  # 角点在三维空间的坐标
    image_points = []  # 角点在图像空间中的位置
    # 生成角点在真实世界中的位置
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # 角点检测
    for file_path in file_paths:
        img = cv2.imread(file_path)
        # 灰度化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 角点检测
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        img_copy = img.copy()
        cv2.drawChessboardCorners(img_copy, (nx, ny), corners, ret)
        plot_contrast_image(img, img_copy)
        if ret:
            object_points.append(objp)
            image_points.append(corners)
    # 相机校正
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

# 图像去畸变：利用相机校正的内参，畸变系数
#def img_undistort(img, mtx, dist):
   # dis = cv2.undistort(img, mtx, dist, None, mtx)
    #return dis
# 图像去畸变：利用相机校正的内参和畸变系数
def img_undistort(img, mtx, dist):
    undistort_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undistort_img


if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = cal_calibrate_params(file_paths)
    if np.all(mtx != None):
        img = cv2.imread("./test/00000.jpg")
        undistort_img = img_undistort(img, mtx, dist)
        plot_contrast_image(img, undistort_img, "Original Image", "Undistorted Image")
        print("done")
    else:
        print("failed")

    # 测试车道线提取
    img = cv2.imread('./test/frame45.jpg')
    result = pipeline(img)
    plot_contrast_image(img, result, converted_img_gray=True)

    # 测试透视变换
    img = cv2.imread('./test/test1.jpg')
    points = [[601, 448], [683, 448], [230, 717], [1097, 717]]
    img = cv2.line(img, (601, 448), (683, 448), (0, 0, 255), 3)
    img = cv2.line(img, (683, 448), (1097, 717), (0, 0, 255), 3)
    img = cv2.line(img, (1097, 717), (230, 717), (0, 0, 255), 3)
    img = cv2.line(img, (230, 717), (601, 448), (0, 0, 255), 3)
    # plt.figure()
    # plt.imshow(img[:, :, ::-1])
    # plt.title("原图")
    # plt.show()
    M, M_inverse = cal_perspective_params(img, points)
    # if np.all(M != None):
    #     trasform_img = img_perspect_transform(img, M)
    #     plt.figure()
    #     plt.imshow(trasform_img[:, :, ::-1])
    #     plt.title("俯视图")
    #     plt.show()
    # else:
    #     print("failed")


    #img = cv2.imread('./test/straight_lines2.jpg')
    # undistort_img = img_undistort(img,mtx,dist)
    # pipeline_img = pipeline(undistort_img)
    # trasform_img = img_perspect_transform(pipeline_img,M)
    # left_fit,right_fit = cal_line_param(trasform_img)
    # result = fill_lane_poly(trasform_img,left_fit,right_fit)
    # plt.figure()
    # plt.imshow(result[:,:,::-1])
    # plt.title("俯视图：填充结果")
    # plt.show()
    # trasform_img_inv = img_perspect_transform(result,M_inverse)
    # plt.figure()
    # plt.imshow(trasform_img_inv[:, :, ::-1])
    # plt.title("填充结果")
    # plt.show()
    # res = cv2.addWeighted(img,1,trasform_img_inv,0.5,0)
    # plt.figure()
    # plt.imshow(res[:, :, ::-1])
    # plt.title("安全区域")
    # plt.show()
    #lane_center = cal_line_center(img)
    #print("中心点位置：{}".format(lane_center))

def process_image(img):
    # 图像去畸变
    undistort_img = img_undistort(img,mtx,dist)
    # 车道线检测
    rigin_pipline_img = pipeline(undistort_img)
    # 透视变换
    transform_img = img_perspect_transform(rigin_pipline_img,M)
    # 拟合车道线
    left_fit,right_fit = cal_line_param(transform_img)





