import cv2
import numpy as np
import argparse
import glob


def draw_cam(input_path, output_path, points):
    for img_path in glob.glob(input_path + "*.png"):

        img = cv2.imread(img_path)
        imgpts = np.int32(points).reshape(-1, 2)

        # draw ground floor in green
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

        cv2.imshow(img_path[len(input_path):], img)
        cv2.waitKey(0)

        out_path = output_path + img_path[len(input_path):]
        cv2.imwrite(out_path, img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input_images", type=str, required=True,
                    help="input images path")

    ap.add_argument("-o", "--output_path", type=str, required=True,
                    help="output path")

    args = ap.parse_args()

    focal_length = 100
    principal_point = (960, 540)

    points_2d = np.load('vr2d.npy')
    points_3d = np.load('vr3d.npy')
    camera_matrix = np.array(
        [[focal_length, 0, principal_point[0]],
         [0, focal_length, principal_point[1]],
         [0, 0, 1]], dtype="double"
    )
    print("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((5, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(points_3d, points_2d, camera_matrix,
                                                                dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))

    cam_axis = np.float32([[0, 0, 0], [0, -3, 0], [-3, -3, 0], [-3, 0, 0],
                          [0, 0, -3], [0, -3, -3], [-3, -3, -3], [-3, 0, -3]])

    points, jac = cv2.projectPoints(cam_axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    draw_cam(args.input_images, args.output_path, points)
