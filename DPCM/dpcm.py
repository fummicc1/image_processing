import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt


def dpcm(img: np.ndarray):
    img = img.astype(np.int16)
    ret = np.zeros(img.shape, dtype=np.int16)
    for channel in range(img.shape[2]):
        for row in range(img.shape[0]):
            for column in range(img.shape[1]):
                if column == 0:
                    continue
                else:
                    ret[row, column, channel] = img[row, column,
                                                    channel] - img[row, column-1, channel]

    cv2.imwrite("out/dpcm_blue.jpg", ret[:, :, 0])
    cv2.imwrite("out/dpcm_green.jpg", ret[:, :, 1])
    cv2.imwrite("out/dpcm_red.jpg", ret[:, :, 2])


def dpcm_gray_scale(img: np.ndarray):
    img = img.astype(np.int16)
    ret = np.zeros(img.shape, dtype=np.int16)
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            if column == 0:
                continue
            else:
                ret[row, column] = img[row, column] - img[row, column-1]

    cv2.imwrite("out/dpcm_gray.jpg", ret)
    return ret


def extract_heads(img: np.ndarray):
    heads = img.copy()
    heads[:, 1:] = 0
    cv2.imwrite("out/dpcm__sendable_heads.jpg", heads)


def restore(diff: np.ndarray, heads):
    ret = np.zeros(diff.shape, dtype=np.uint8)
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            if column == 0:
                ret[row, column] = heads[row]
            else:
                ret[row, column] = ret[row, column-1] + diff[row, column]

    cv2.imwrite("out/dpcm_restore.jpg", ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path")
    args = parser.parse_args()
    img_path = "img/Lenna.png"
    if args.img_path:
        img_path = args.img_path
    img = cv2.imread(img_path)
    dpcm(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff = dpcm_gray_scale(img)
    cv2.imwrite("out/gray_base.jpg", img)
    restore(diff, img[:, 0])
    extract_heads(img)
