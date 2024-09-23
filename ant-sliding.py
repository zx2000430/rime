import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh


def perspective_transform(img):
    """
    Execute perspective transform
    """
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[200, 720],
         [1100, 720],
         [595, 450],
         [685, 450]])
    dst = np.float32(
        [[300, 720],
         [980, 720],
         [300, 0],
         [980, 0]])

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)

    # 获取图像宽度和高度
    width = warped.shape[1]
    height = warped.shape[0]

    # 定义左右车道线的宽度和间隔
    lane_width = 100  # 左右车道线的宽度，根据实际情况调整
    lane_gap = 150  # 左右车道线之间的间隔，根据实际情况调整

    # 定义左右车道线的边界
    left_lane_boundary = width // 2 - lane_width - lane_gap // 2
    right_lane_boundary = width // 2 + lane_gap // 2

    # 将图像分为左右两块
    left_block = warped[:, :left_lane_boundary]
    right_block = warped[:, right_lane_boundary:]

    return warped, unwarped, m, m_inv, left_block, right_block

def find_lane_start_end(warped_left, warped_right):
    # 左车道线的顶部扫描
    for row in range(warped_left.shape[0]):
        if np.any(warped_left[row, :]):
            left_end_point = (row, np.argmax(warped_left[row, :]))
            break

    # 右车道线的顶部扫描
    for row in range(warped_right.shape[0]):
        if np.any(warped_right[row, :]):
            right_end_point = (row, np.argmax(warped_right[row, :]))
            break

    # 左车道线的底部扫描
    for row in range(warped_left.shape[0] - 1, -1, -1):
        if np.any(warped_left[row, :]):
            left_start_point = (row, np.argmax(warped_left[row, :]))
            break

    # 右车道线的底部扫描
    for row in range(warped_right.shape[0] - 1, -1, -1):
        if np.any(warped_right[row, :]):
            right_start_point = (row, np.argmax(warped_right[row, :]))
            break

    return left_start_point, left_end_point, right_start_point, right_end_point

def ant_colony_optimization(warped_left, warped_right):
    # 初始化信息素分布矩阵
    pheromone = np.ones(warped_left.shape)

    # 设置蚁群算法参数
    num_ants = 10  # 蚂蚁数量
    num_iterations = 100  # 迭代次数
    evaporation_rate = 0.5  # 信息素蒸发率
    alpha = 1.0  # 信息素重要程度因子
    beta = 2.0  # 启发式信息重要程度因子

    # 调用 find_lane_start_end 函数获取左右车道线的起点和终点坐标
    left_start_point, left_end_point, right_start_point, right_end_point = find_lane_start_end(warped_left, warped_right)

    # 迭代搜索过程
    for iteration in range(num_iterations):
        left_ant_paths = []
        right_ant_paths = []

        # 左边路径搜索
        for ant in range(num_ants):
            current_pixel = left_start_point
            path = [current_pixel]

            while current_pixel != left_end_point:
                current_pixel = path[-1]
                next_pixel = None

                # 检查当前像素位置是否合法
                if current_pixel[0] >= pheromone.shape[0]:
                    break

                # 计算下一个像素位置的概率
                probabilities = np.power(np.asarray(pheromone[current_pixel]), alpha)
                probabilities = np.atleast_1d(probabilities)
                indices = np.arange(probabilities.shape[0])
                probabilities = np.where(indices == current_pixel[1], 0.0, probabilities)
                probabilities /= np.sum(probabilities)

                # 创建临时概率列表
                temp_probabilities = probabilities.tolist()

                # 创建与范围相同长度的数组
                indices = np.arange(warped_left.shape[1])

                if len(temp_probabilities) == len(indices):
                    # 进行随机选择
                    next_pixel = np.random.choice(indices, p=temp_probabilities)
                else:
                    # 处理长度不匹配的情况
                    # 或者选择默认行为，如均匀随机选择
                    next_pixel = np.random.choice(indices)
                # 轮盘赌选择下一个像素位置
                #next_pixel = np.random.choice(range(warped_left.shape[1]), p=temp_probabilities)

                # 更新路径
                path.append((current_pixel[0] + 1, next_pixel))
                current_pixel = (current_pixel[0] + 1, next_pixel)

            left_ant_paths.append(path)

        # 右边路径搜索
        for ant in range(num_ants):
            current_pixel = right_start_point
            path = [current_pixel]

            while current_pixel != right_end_point:
                current_pixel = path[-1]
                next_pixel = None

                # 检查当前像素位置是否合法
                if current_pixel[0] >= pheromone.shape[0]:
                    break

                # 计算下一个像素位置的概率
                probabilities = np.power(np.asarray(pheromone[current_pixel]), alpha)
                probabilities = np.atleast_1d(probabilities)
                indices = np.arange(probabilities.shape[0])
                probabilities = np.where(indices == current_pixel[1], 0.0, probabilities)
                probabilities /= np.sum(probabilities)

                # 创建临时概率列表
                temp_probabilities = probabilities.tolist()

                # 创建与范围相同长度的数组
                indices = np.arange(warped_right.shape[1])

                if len(temp_probabilities) == len(indices):
                    # 进行随机选择
                    next_pixel = np.random.choice(indices, p=temp_probabilities)
                else:
                    # 处理长度不匹配的情况
                    # 或者选择默认行为，如均匀随机选择
                    next_pixel = np.random.choice(indices)

                # 更新路径
                path.append((current_pixel[0] + 1, next_pixel))
                current_pixel = (current_pixel[0] + 1, next_pixel)

            right_ant_paths.append(path)

        # 更新信息素
        delta_pheromone_left = np.zeros(warped_left.shape)
        delta_pheromone_right = np.zeros(warped_right.shape)
        #delta_pheromone = np.zeros(warped_left.shape)

        for left_path, right_path in zip(left_ant_paths, right_ant_paths):
            # 更新左路径的信息素
            for i in range(len(left_path) - 1):
                current_pixel = left_path[i]
                next_pixel = left_path[i + 1]

                delta_pheromone_left[current_pixel[0], next_pixel[1]] += 1.0

            # 更新右路径的信息素
            for i in range(len(right_path) - 1):
                current_pixel = right_path[i]
                next_pixel = right_path[i + 1]

                delta_pheromone_right[current_pixel[0], next_pixel[1]] += 1.0

        pheromone_left = (1.0 - evaporation_rate) * pheromone_left + delta_pheromone_left
        pheromone_right = (1.0 - evaporation_rate) * pheromone_right + delta_pheromone_right

    # 选择最优路径
    best_left_path = None
    best_right_path = None
    max_left_pheromone = 0
    max_right_pheromone = 0
    print("left_path:", left_path)
    print("right_path:", right_path)
    print("pheromone shape:", pheromone.shape)
    left_path = np.array(left_path)
    n, m = pheromone.shape
    left_path[:, 0] = left_path[:, 0] % n  # 对行索引进行修正
    left_path[:, 1] = left_path[:, 1] % m  # 对列索引进行修正

    left_pheromone = np.sum(pheromone[left_path[:, 0], left_path[:, 1]])
    right_path = np.array(right_path)
    n, m = pheromone.shape
    right_path[:, 0] = right_path[:, 0] % n  # 对行索引进行修正
    right_path[:, 1] = right_path[:, 1] % m  # 对列索引进行修正

    right_pheromone = np.sum(pheromone[right_path[:, 0], right_path[:, 1]])

    for left_path, right_path in zip(left_ant_paths, right_ant_paths):
        left_path = np.array(left_path)
        left_pheromone = np.sum(pheromone[left_path[:, 0], left_path[:, 1]])
        right_path = np.array(right_path)
        right_pheromone = np.sum(pheromone[right_path[:, 0], right_path[:, 1]])

        if left_pheromone > max_left_pheromone:
            max_left_pheromone = left_pheromone
            best_left_path = left_path

        if right_pheromone > max_right_pheromone:
            max_right_pheromone = right_pheromone
            best_right_path = right_path

    # 返回最佳左路和右路路径
    return best_left_path, best_right_path

    # 返回最佳路径和信息素分布矩阵
    #return best_path, pheromone

def cal_line_param(binary_warped):
    # 1.确定左右车道线的位置
    # 统计直方图
    histogram = np.sum(binary_warped[:, :], axis=0)
    # 在统计结果中找到左右最大的点的位置，作为左右车道检测的开始点
    # 将统计结果一分为二，划分为左右两个部分，分别定位峰值位置，即为两条车道的搜索位置
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # 2.滑动窗口检测车道线
    # 设置滑动窗口的数量，计算每一个窗口的高度
    nwindows = 9
    window_height = int(binary_warped.shape[0] / nwindows)
    # 获取图像中不为0的点
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # 车道检测的当前位置
    leftx_current = leftx_base
    rightx_current = rightx_base
    # 设置x的检测范围，滑动窗口的宽度的一半，手动指定
    margin = 60
    # 设置最小像素点，阈值用于统计滑动窗口区域内的非零像素个数，小于50的窗口不对x的中心值进行更新
    minpix = 50
    # 用来记录搜索窗口中非零点在nonzeroy和nonzerox中的索引
    left_lane_inds = []
    right_lane_inds = []

    # 绘制统计直方图
    plt.figure(3)
    plt.plot(histogram)
    plt.title("统计直方图")
    #plt.show()

    # 遍历该副图像中的每一个窗口
    for window in range(nwindows):
        # 设置窗口的y的检测范围，因为图像是（行列）,shape[0]表示y方向的结果，上面是0
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # 左车道x的范围
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        # 右车道x的范围
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 确定非零点的位置x,y是否在搜索窗口中，将在搜索窗口内的x,y的索引存入left_lane_inds和right_lane_inds中
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 如果获取的点的个数大于最小个数，则利用其更新滑动窗口在x轴的位置
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # 将检测出的左右车道点转换为array
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 获取检测出的左右车道点在图像中的位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 绘制滑动窗口拟合效果图
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # 绘制左车道线的窗口
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        cv2.rectangle(window_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    # 绘制右车道线的窗口
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(window_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
    # 将窗口图像叠加到原始图像上
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # 显示滑动窗口拟合效果图
    plt.figure(4)
    plt.imshow(result , cmap='gray')
    plt.title("滑动窗口拟合效果图")
    plt.show()

    # 3.用曲线拟合检测出的点,二次多项式拟合，返回的结果是系数
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

if __name__ == '__main__':
    img_file = 'test_images/straight_lines1.jpg'

    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    img = mpimg.imread(img_file)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)
    # 执行透视变换和二值化处理
    warped, unwarped, m, m_inv, left_block, right_block = perspective_transform(img)

    # 创建第一个图像窗口并显示warped图像
    plt.figure(1)
    plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
    plt.title('Warped Image')
    # 创建第二个图像窗口并显示unwarped图像
    plt.figure(2)
    plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
    plt.title('Unwarped Image')
    # 创建第三个图像窗口并显示left_block图像
    plt.figure(3)
    plt.imshow(left_block, cmap='gray', vmin=0, vmax=1)
    plt.title('Left Block')
    # 创建第四个图像窗口并显示right_block图像
    plt.figure(4)
    plt.imshow(right_block, cmap='gray', vmin=0, vmax=1)
    plt.title('Right Block')

    # 寻找起点和终点
    left_start_point, left_end_point, right_start_point, right_end_point = find_lane_start_end(left_block, right_block)

    print("左车道线顶部点：", left_end_point)
    print("右车道线顶部点：", right_end_point)
    print("左车道线底部点：", left_start_point)
    print("右车道线底部点：", right_start_point)

    # 将 left_block 和 right_block 水平合并
    image = np.hstack((left_block, right_block))
    best_path = ant_colony_optimization(left_block, right_block)
    best_left_path = np.array(best_path[0])
    best_right_path = np.array(best_path[1])
    print("最优左路路径：", best_left_path)
    print("最优右路路径：", best_right_path)

    # 显示最优路径
    plt.figure(5)
    plt.imshow(image, cmap='gray')
    plt.plot([pixel[1] for pixel in best_path], [pixel[0] for pixel in best_path], color='red')
    plt.title('best_path')
    plt.show()

    # 显示信息素分布
    plt.figure(6)
    plt.imshow(pheromone, cmap='hot')
    plt.title('pheromone')
    plt.colorbar()
    #plt.show()

plt.show()

left_fit, right_fit = cal_line_param(warped)

