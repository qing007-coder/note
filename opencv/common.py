import cv2
import numpy as np
from typing import Sequence, Tuple


class Opencv:

    def __init__(self, path: str):
        self.img = cv2.imread(path)
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

    @classmethod
    def threshold_binary(cls, img: np.ndarray) -> np.ndarray:
        """
        先转成灰度图 然后再二值化
        流程是把三通道的图转成单通道（即灰度图）
        127是阈值 如果灰度图中的像素值大于 127，就变成 255（纯白） 如果像素值小于或等于 127，就变成 0（纯黑）
        :param img:
        :return:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return threshold

    @classmethod
    def adaptive_threshold(cls, img: np.ndarray) -> np.ndarray:
        """
        也是先转灰度图 然后进行自适应
        C 设得太小（比如 0 或负数）： 图像对噪声极度敏感。背景会变得很脏，充满密密麻麻的黑色噪点。C 设得太大（比如 20、30）： 门槛提得太高。虽然背景极其干净，但原本一些颜色比较浅的文字、细线条、或者微弱的特征，也会直接被当成背景抹杀掉（文字变秃、线条断裂）。
        :param img:
        :return:
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,  # 1. 输入图
            255,  # 2. 最大值
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 3. 自适应方法
            cv2.THRESH_BINARY,  # 4. 阈值类型
            11,  # 5. 区域大小 (Block Size) 11*11范围
            2  # 6. 常数 C 噪声容忍度
        )
        return binary

    @classmethod
    def canny_edges(cls, img: np.ndarray) -> np.ndarray:
        """
        用于边缘检测
        比例关系： 经验上，低阈值和高阈值的比例通常建议设在 1:2 到 1:3 之间（比如 50, 150 或 30, 90）。
        把数字整体调小（如 20, 60）： 算法变得非常敏感。原本不是边缘的微小纹理、甚至图片上的杂质、噪点都会被当成边缘画出来，线条会变多、变杂乱。
        把数字整体调大（如 100, 300）： 算法变得非常严格。只有对比度极高、极其明显的边界才会被画出来，很多弱一点的线条会断裂、甚至直接消失。
        :param img:
        :return:
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(
            gray,  # 1. 输入图
            50,  # 2. 低阈值 (minVal)
            150  # 3. 高阈值 (maxVal)
        )

        return edge

    @classmethod
    def find_contours(cls, edge: np.ndarray) -> Sequence[np.ndarray]:
        contours, _ = cv2.findContours(
            edge,  # 1. 输入图像
            cv2.RETR_EXTERNAL,  # 2. 轮廓检索模式
            cv2.CHAIN_APPROX_SIMPLE  # 3. 轮廓近似方法
        )

        return contours

    @classmethod
    def filter_by_area(cls, contours: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """
        计算目标区域来进行噪声过滤
        :param contours:
        :return:
        """
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:    # 过滤噪声
                filtered_contours.append(contour)
        return filtered_contours

    @classmethod
    def draw_rectangle(cls, img: np.ndarray, contours: Sequence[np.ndarray]) -> None:
        """
        将轮廓位置用矩形圈出
        :param img:
        :param contours:
        :return:
        """
        drawn_img = img.copy()
        # 核心：通过 for 循环，把里面的每一个单个轮廓 (contour) 依次拿出来计算
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)  # 此时传的是单个，不会报错了
            cv2.rectangle(
                drawn_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # 循环画完所有框后，再统一显示
        Opencv.show_img(drawn_img)

    @classmethod
    def dilate(cls, edge: np.ndarray, params: Tuple[int, int], iteration: int) -> np.ndarray:
        """
        图像膨胀：让白色区域向外扩张（类似于 | 的关系）。
        只要毛刷覆盖的区域内有至少一个白点(255)，中心点就变成白点。
        作用：桥接断裂的边缘、融合线条。
        """
        kernel = np.ones(params, np.uint8)
        return cv2.dilate(edge, kernel, iterations=iteration)

    @classmethod
    def erode(cls, edge: np.ndarray, params: Tuple[int, int], iteration: int) -> np.ndarray:
        """
        图像腐蚀：让白色区域向内坍缩（类似于 & 的关系）。
        只有毛刷覆盖的区域内全都是白点(255)，中心点才保持白色；否则变黑。
        作用：消除孤立的微小白色噪点、切断物体间细微的粘连。
        """
        kernel = np.ones(params, np.uint8)
        return cv2.erode(edge, kernel, iterations=iteration)

    @classmethod
    def morphology_ex(cls, edge: np.ndarray, op_type: int, params: Tuple[int, int]) -> np.ndarray:
        """
        形态学高级操作（开运算/闭运算）。
        :param edge:
        :param op_type: cv2.MORPH_OPEN (开运算) 或 cv2.MORPH_CLOSE (闭运算)
        :param params: 算子的长宽参数，如 (3, 3)
        """
        kernel = np.ones(params, np.uint8)
        return cv2.morphologyEx(edge, op_type, kernel)

    def pipeline(self) -> None:
        """
        标准处理流水线 流程 转灰度图 -> 找边缘 -> 边缘更明显 -> 边缘转轮廓 -> 过滤噪声轮廓
        :return:
        """

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 50, 150)

        kernel = np.ones((3, 3), np.uint8)
        # eroded = cv2.erode(edge, kernel, iterations=1)
        morphology_edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("edge", morphology_edge)

        contours, _ = cv2.findContours(morphology_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        drawn_img = self.img.copy()
        print(len(contours))
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(drawn_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Cleaned Object Detection", drawn_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
