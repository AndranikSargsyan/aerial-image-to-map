import argparse

import cv2
from pylsd import lsd
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, required=True, help='Path to image.')
    return parser.parse_args()


def main():
    args = get_args()

    img = cv2.imread(args.img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    segments = lsd(img_gray, scale=0.8, sigma_scale=1.0, ang_th=40, density_th=0.01)

    for i in range(segments.shape[0]):
        pt1 = (int(segments[i, 0]), int(segments[i, 1]))
        pt2 = (int(segments[i, 2]), int(segments[i, 3]))
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)

    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
