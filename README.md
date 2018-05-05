# calibration_py

## The calibresult.py is a process used for calibration and Distortion of camera.

#### Create a directory like "left" and put pictures inside, modify the corresponding path name and run it, you will get the intrinsic matrix from the value mtx, get the extrinsic matrix's rotation and translation parameters each picture from rvecs and tvecs, get the distortion coefficients from dist, and the modified picture in calibresult01.png. If you want, you can just print these.

## The calibrate_zhang.py is a process to implement zhang's method.

#### It uses DLT method to get the H, cholesky decomposition to get the intrinsic matrix and extrinsic matrix.
#### As calibresult.y above, config its path and just run it, you will get the H, the A, and cost successively.
#### However, it has a very tricky problem to optimize B, because of the blind initialization and uncontrollable update. It achieves 180 cost in the first image while the library function gets around 1. Hope to receive good advice.
