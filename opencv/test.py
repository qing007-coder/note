from .common import Opencv


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

    Opencv.start_camera()
