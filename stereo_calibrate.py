import cv2
import numpy as np
import glob

def get_parameter(images, w, h):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        if ret == True:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            continue

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def cal1():
    w = 9
    h = 6
    size = (640, 480)
    images_left = glob.glob('./left/*.jpg')
    images_right = glob.glob('./right/*.jpg')
    #获得相机参数
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = get_parameter(images_left, w, h)
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = get_parameter(images_right, w, h)

    #相对R和T的计算
    R = np.matmul(cv2.Rodrigues(np.array(rvecs_r[0]))[0], np.linalg.inv(cv2.Rodrigues(np.array(rvecs_l[0]))[0]))
    T = np.reshape(np.array(tvecs_r[0]) - np.matmul(R, np.array(tvecs_l[0])), [3])
    mtx_l = np.array(mtx_l)
    dist_l = np.array(dist_l)
    mtx_r = np.array(mtx_r)
    dist_r = np.array(dist_r)

    #立体更正
    R1, R2, P1, P2, Q, ROI1, ROI2 =  cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, size, R, T)
    #计算更正map
    left_map1, left_map2 = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, size, cv2.CV_16SC2)

    img1 = cv2.imread('./left/left01.jpg')
    img2 = cv2.imread('./right/right01.jpg')
    #map图像
    img1_rect = cv2.remap(img1, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rect = cv2.remap(img2, right_map1, right_map2, cv2.INTER_LINEAR)

    imgL = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY)
    #生成深度图
    stereo = cv2.StereoBM_create(numDisparities=16 * 10, blockSize=11)
    disparity = stereo.compute(imgL, imgR)
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow("left", img1_rect)
    cv2.imshow("right", img2_rect)
    cv2.imshow("depth", disp)
    cv2.waitKey(0)


if __name__ == '__main__':
    cal1()