import cv2
import numpy as np
import glob
from scipy.optimize import minimize

def cal_V(v1, v2):
    return np.array([v1[0]*v2[0], v1[0]*v2[1] + v1[1]*v2[0], v1[1]*v2[1], v1[0]*v2[2] + v1[2]*v2[0],
            v1[2]*v2[1] + v1[1] * v2[2], v1[2] * v2[2]])


def err(b, w):
    K = abs(np.matmul(w, b.T))
    K /= 1e4 # 尽可能提前停止，否则全是0
    return sum(K, 0)

def err1(b, w):
    K = abs(np.matmul(w, b.T)) * 1e15
    return sum(K, 0)

def cal1():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w = 9
    h = 6
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    images = glob.glob('./left/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        if ret == True:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            exit(-1)

    # my implement
    obj = (np.array(objpoints))[:,:,0:-1]
    ori = np.array(imgpoints).reshape([13, 54, 2])
    H = []
    tmp = np.ones([w * h])
    #每张图片求H，result为每张图片H的结果
    for i in range(len(images)):
        cur_obj = obj[i]
        cur_ori = ori[i]
        cur_obj = np.column_stack((cur_obj, tmp)).T
        cur_ori = np.column_stack((cur_ori, tmp)).T
        result = np.matmul(np.matmul(cur_ori, cur_obj.T), np.linalg.inv(np.matmul(cur_obj, cur_obj.T)))
        H.append(result)

    print(result)

    H = np.array(H)
    B = np.ones([6], dtype = float) * 5e3
    T = []
    for i in range(len(H)):
        h1 = H[i][:, 0]
        h2 = H[i][:, 1]
        # h3 = H[i][:, 3]
        v12 = cal_V(h1, h2)
        v11_v22 = cal_V(h1, h1) - cal_V(h2, h2)
        T.append(v12)
        T.append(v11_v22)
    T = np.array(T, dtype = float)
    #最小化求B
    B = minimize(err, B, (T), method='COBYLA').x
    # 让后3个参数得到更新，但这对cost没有丝毫影响
    B[3:6] = minimize(err1, B[3:6], (T[:,3:6]), method='COBYLA').x
    #cholesky分解求内参
    T_B = np.array([[B[0], B[1], B[3]],[B[1], B[2], B[4]], [B[3], B[4], B[5]]])
    L = np.linalg.cholesky(T_B)
    C = np.linalg.inv(L.T)
    print(C)

    # 这是论文提供的求内参的方法
    # v0 = (B[1] * B[3] - B[0] * B[4]) / (B[0] * B[2] - B[1] * B[1])
    # lamd = B[5] - (B[3] * B[3] + v0 * (B[1] * B[3] - B[0] * B[4])) / B[0]
    # alpha = math.sqrt(lamd / B[0])
    # beta = math.sqrt(lamd * B[0] / (B[0] * B[2] - B[1] * B[1]))
    # gam = -B[1] * alpha * alpha * beta / lamd
    # u0 = gam * v0 / alpha - B[3] * alpha * alpha / lamd
    # print(v0, lamd, alpha, beta, gam, u0)

    # 论文提供求外参方法
    H = np.array(H[0])
    A = np.array(C)
    a_h1 = np.dot(np.linalg.inv(A), H[:, 0])
    lamd = 1 / np.linalg.norm(a_h1, ord=2)
    r1 = lamd * a_h1
    r2 = lamd * np.dot(np.linalg.inv(A), H[:, 1])
    r3 = np.cross(r1, r2)
    t = lamd * np.dot(np.linalg.inv(A), H[:, 2])
    RT = np.array([r1,r2,r3,t]).T

    #一张图片的cost
    cost = 0.0
    tmp2 = np.zeros([w*h, 2])
    tmp2[:,1] = tmp
    cur_obj = obj[0]
    cur_ori = ori[0]
    cur_obj = np.column_stack((cur_obj, tmp))
    cur_ori = np.column_stack((cur_ori, tmp2))
    cost += sum(sum(abs(np.matmul(np.matmul(A, RT), cur_ori.T) - cur_obj.T), 0), 0)
    print(cost)


if __name__ == '__main__':
    cal1()