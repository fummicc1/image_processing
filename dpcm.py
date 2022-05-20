import cv2
import numpy as np
import argparse


def dpcm(img: np.ndarray):
    print(img.shape)
    img = img.astype(np.int16)
    ret = np.zeros(img.shape, dtype=np.int16)
    for channel in range(img.shape[2]):
        print(f"channel: {channel}")
        for row in range(img.shape[0]):
            for column in range(img.shape[1]):
                pixel = img[row, column, channel]
                if column == 0:
                    continue
                else:
                    ret[row, column, channel] = img[row, column,
                                                    channel] - img[row, column-1, channel]
                    if img[row, column, channel] - img[row, column-1, channel] >= 200:
                        print(f"img: {img[row, column, channel]}")
                        print(f"previous img: {img[row, column-1, channel]}")
                        print(
                            f"diff: {img[row, column, channel]-img[row, column-1, channel]}")

    ret = np.array(ret)
    print(len(ret[ret[:, :, :] == 0]))
    print(ret.shape)
    cv2.imwrite("out/dpcm_blue.jpg", ret[:, :, 0])
    cv2.imwrite("out/dpcm_green.jpg", ret[:, :, 1])
    cv2.imwrite("out/dpcm_red.jpg", ret[:, :, 2])


def dpcm_gray_scale(img: np.ndarray):
    print(img.shape)
    img = img.astype(np.int16)
    ret = np.zeros(img.shape, dtype=np.int16)
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            pixel = img[row, column]
            if column == 0:
                continue
            else:
                ret[row, column] = img[row, column] - img[row, column-1]

    ret = np.array(ret)
    cv2.imwrite("out/dpcm_gray.jpg", ret)


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
    dpcm_gray_scale(img)
