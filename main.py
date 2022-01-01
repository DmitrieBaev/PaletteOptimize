import pathlib, sys, cv2, numpy as np


def quantization(img, k):
    """Оптимизирует палитру изображения

    :param img:
    :param k: int
        Количество цветов в новой палитре.
    :return:
        Изображение с оптимизированной палитрой.
    """

    data = np.float32(img).reshape((-1, 3))
    # Внедряем метод k-средних
    ret, label, center = cv2.kmeans(data, k, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001),
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    return center[label.flatten()].reshape(img.shape)


if __name__ == '__main__':
    input_img = cv2.imread(pathlib.Path(sys.argv[1]))
    kcolor = int(sys.argv[2])
    output_img = quantization(input_img, kcolor)
    cv2.imwrite(f'quantized_{kcolor}.jpg', output_img)
