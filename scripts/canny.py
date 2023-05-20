import argparse

import cv2
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, required=True, help='Path to image.')
    return parser.parse_args()


def main():
    args = get_args()

    img = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    edges = cv2.Canny(img, 100, 200, apertureSize=3)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    main()
