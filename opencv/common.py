import cv2
import numpy as np


class Opencv:

    def __init__(self) -> None:
        return

    @classmethod
    def read_img(cls, path: str) -> np.ndarray:
        return cv2.imread(path)

    @classmethod
    def show_img(cls, img: np.ndarray) -> None:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @classmethod
    def show_gray_img(cls, img: np.ndarray) -> np.ndarray:
        """
        灰度图
        :param img:
        :return:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return gray

    @classmethod
    def in_range(cls, img: np.ndarray, low: tuple, high: tuple) -> np.ndarray:
        """
        过滤条件
        :param img:
        :param low:
        :param high:
        :return:
        """
        return cv2.inRange(img, low, high)

    @classmethod
    def convert_hsv(cls, img: np.ndarray) -> np.ndarray:
        """
        转成hsv格式图片
        :param img:
        :return:
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    @classmethod
    def resize(cls, img: np.ndarray, params: tuple) -> np.ndarray:
        """
        更改图片大小
        :param img:
        :param params:
        :return:
        """
        return cv2.resize(img, params)

    @classmethod
    def start_camera(cls):
        while True:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()

            if not ret:
                break

            hsv = Opencv.convert_hsv(frame)
            lower = (0, 120, 70)
            upper = (10, 255, 255)
            frame = Opencv.in_range(hsv, lower, upper)
            cv2.imshow(
                "camera",
                frame
            )

            if cv2.waitKey(1) == 27:
                break
