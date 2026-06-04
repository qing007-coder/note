from .common import Opencv
import cv2


def test_opencv():
    img = Opencv.read_img("C:\\Users\\32716\\Desktop\\node\\4.2.png")
    print(img.shape)

    # Opencv.show_img(img[100:800, 100:800])
    print(img[100, 200])

    # gray_img = Opencv.show_gray_img(img)
    # print(gray_img.shape)

    # lower = (0, 120, 70)
    # upper = (10, 255, 255)
    #
    # hsv = Opencv.convert_hsv(img)
    #
    # Opencv.show_img(hsv)
    #
    # mask = Opencv.in_range(hsv, lower, upper)
    # print(mask.shape)
    #
    # Opencv.show_img(mask)
    #
    # resized_img = Opencv.resize(mask, (500, 500))
    # Opencv.show_img(resized_img)

    # Opencv.start_camera()

    # binary_threshold = Opencv.threshold_binary(img)
    # Opencv.show_img(binary_threshold)

    # adaptive_threshold = Opencv.adaptive_threshold(img)
    # Opencv.show_img(adaptive_threshold)

    edge = Opencv.canny_edges(img)
    Opencv.show_img(edge)
    print(edge.shape)

    # contours = Opencv.find_contours(edge)
    # filtered_contours = Opencv.filter_by_area(contours)
    # contours_img = cv2.drawContours(
    #     img,  # 1. 在哪张图上画
    #     filtered_contours,  # 2. 要画的轮廓数据
    #     -1,  # 3. 画哪一个轮廓
    #     (0, 255, 0),  # 4. 颜色 (BGR)
    #     2  # 5. 线条粗细
    # )
    #
    # Opencv.show_img(contours_img)
    # print(len(contours))
    # print(len(filtered_contours))
    #
    # Opencv.draw_rectangle(img, filtered_contours)

    # dilate_img = Opencv.dilate(edge, (3, 3), iteration=1)
    # Opencv.show_img(dilate_img)
    # print(dilate_img.shape)

    img1 = Opencv("C:\\Users\\32716\\Desktop\\node\\4.2.png")
    img1.pipeline()
