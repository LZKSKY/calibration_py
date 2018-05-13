# calibration_py

## 1.The **calibresult.py** is a process used for calibration and Distortion of camera.

#### Create a directory like "left" and put pictures inside, modify the corresponding path name and run it, you will get the intrinsic matrix from the value mtx, get the extrinsic matrix's rotation and translation parameters each picture from rvecs and tvecs, get the distortion coefficients from dist, and the modified picture in calibresult01.png. If you want, you can just print these.

## 2.The **calibrate_zhang.py** is a process to implement zhang's method.

#### It uses DLT method to get the H, cholesky decomposition to get the intrinsic matrix and extrinsic matrix.
#### As calibresult.y above, config its path and just run it, you will get the H, the A, and cost successively.
#### However, it has a very tricky problem to optimize B, because of the blind initialization and uncontrollable update. It achieves 180 cost in the first image while the library function gets around 1. Hope to receive good advice.

## 3.The **stereo_calibrate.py** is a process for stero calibration.

#### Left camera and right camera's pictures are respectively in './left/' and './right/' with size of (640, 480). It first gets each camera's intrinsic matrix, rotation matrix and translation matrix from camera calibration with the same method as **calibresult.py**. Then, map image to its stereo calibration's result and use SGM to calculate the disparity result with left and right images.
