import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os

def perspective_transform(img):
    """
    Execute perspective transform
    """
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[200, 600],
         [1000, 600],
         [790, 300],
         [840, 300]])
    dst = np.float32(
        [[300, 600],
         [1200, 600],
         [850, 90],
         [1200, 90]])

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, unwarped, m, m_inv


if __name__ == '__main__':
    img_file = 'test_images/00000.jpg'

    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    img = mpimg.imread(img_file)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    warped, unwarped, m, m_inv = perspective_transform(img)

    # 指定保存图像的文件夹路径
    save_dir = 'F:/rime'

    # 确保文件夹存在，如果不存在则创建
    os.makedirs(save_dir, exist_ok=True)
    # 创建第一个图像窗口并显示warped图像

    plt.figure(1)
    plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
    plt.title('Warped Image')


    # 保存图像到文件夹中
    #save_path = os.path.join(save_dir, 'Warped.jpg')
    #cv2.imwrite(save_path, warped)

    # 创建第二个图像窗口并显示unwarped图像
    plt.figure()
    plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
    plt.title('Unwarped Image')
    plt.show()