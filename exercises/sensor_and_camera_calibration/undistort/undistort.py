# Cpoyright 2022 aram_father@naver.com

import argparse
import os
import sys
import glob
import cv2
import numpy as np
from typing import List, Optional, Tuple


CHECKER_SIZE = (8, 6)


def parse_command_line_arguments(args: List[str]) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments.

    Returns:
        : Parsed arguments.
    
    """
    parser = argparse.ArgumentParser(
        prog="undistort",
        description="Undistort input image through calibration images."
    )

    parser.add_argument(
        "--calibration-image-directory",
        type=str,
        required=True,
        help="Directory that has calibration image files(GOPR*.jpg)."
    )

    parser.add_argument(
        "--output-directory",
        type=str,
        required=True,
        help="Output file directory."
    )

    parser.add_argument(
        "--input-image",
        type=str,
        required=True,
        help="Input image file path."
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]]):
    """Main.
    
    Args:
        args: Command line arguments.
    """
    args = parse_command_line_arguments(args)

    # Prepare object point.
    obj_point = np.zeros((CHECKER_SIZE[0] * CHECKER_SIZE[1], 3), np.float32)
    obj_point[:,:2] = np.mgrid[0:CHECKER_SIZE[0],0:CHECKER_SIZE[1]].T.reshape(-1, 2)
    
    # Prepare object/image points.
    obj_points = []
    img_points = []

    for cal_img_path in glob.glob(f"{args.calibration_image_directory}/GO*.jpg"):
        cal_img = cv2.imread(cal_img_path)
        cal_img_gray = cv2.cvtColor(cal_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(cal_img_gray, CHECKER_SIZE, None)
        if ret:
            obj_points.append(obj_point)
            img_points.append(corners)
            cv2.drawChessboardCorners(cal_img, CHECKER_SIZE, corners, ret)
            dst_name = os.path.basename(cal_img_path) + "_checker_board.jpg"
            cv2.imwrite(os.path.join(args.output_directory, dst_name), cal_img)

    # Calibrate.
    test_img = cv2.imread(args.input_image)
    test_img_size = (test_img.shape[1], test_img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, test_img_size, None, None)

    # Undistort.
    undist_img = cv2.undistort(test_img, mtx, dist, None, mtx)
    dst_name = os.path.basename(args.input_image) + "_undistorted.jpg"
    cv2.imwrite(os.path.join(args.output_directory, dst_name), undist_img)
    

if __name__ == "__main__":
    main(sys.argv[1:])