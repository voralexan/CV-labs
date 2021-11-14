import cv2
import numpy as np

if __name__ == "__main__":

   ###########################
   # 1. Прочитать изображение из файла.
   ###########################
    img = cv2.imread("image.png")
    height, width, channels = img.shape

   ###########################
   # 2. Преобразовать изображение в YUV. Далее для Y – канала выполнить шаги 3-8.
   ###########################

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

   ###########################
   # 3. Улучшить контраст (equalizeHist).
   ###########################

    eq_img = cv2.equalizeHist(y)

   ###########################
   # 4. Найти края объектов (Canny).
   ###########################

    img_canny = cv2.Canny(eq_img,150,150)

   ###########################
   # 5. Найти угловые точки на изображении. Нарисовать их кругом с радиусом r (r=2) в тоже изображение где края.
   ###########################
    corner_image = cv2.cornerHarris(img_canny, 2, 3, 0.1)

    corner_image[corner_image > (0.01 * corner_image.max())] = 255

    circled_corner = np.zeros(y.shape, dtype="ubyte")
    for j in range(width):
        for i in range(height):
            if corner_image[i, j] > 250:
                cv2.circle(circled_corner, (j, i), 2, (255, 0, 0), -1)

    corner_image = img_canny + circled_corner
   ###########################
   # 6.  Для найденных границ и угловых точек строится карта расстояний D[i,j] методом distance transform.
   ###########################
    distance_image = cv2.distanceTransform(255 - corner_image, cv2.DIST_L2, 3)
    cv2.normalize(distance_image, distance_image, 0, 1.0, cv2.NORM_MINMAX)
   ###########################
   # 7,8.  В каждом пикселе [i,j] производится фильтрация усреднением.
   ###########################

    integral = cv2.integral(y)
    averaging = np.zeros(y.shape, dtype="ubyte")
    xs = np.arange(y.shape[0], dtype='int32')
    ys = np.arange(y.shape[1], dtype='int32')
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    k = 70
    rad = (k * distance_image).astype('int32') + 1

    x_left = np.maximum(X - rad, 0)
    x_right = np.minimum(X + rad, y.shape[0] - 1)
    y_left = np.maximum(Y - rad, 0)
    y_right = np.minimum(Y + rad, y.shape[1] - 1)

    for i in range(height):
        for j in range (width):
            for ch in range(channels):
                x_l = x_left[i][j]
                x_r = x_right[i][j]
                y_l = y_left[i][j]
                y_r = y_right[i][j]

                A = integral[x_l][y_l]
                B = integral[x_r][y_l]
                C = integral[x_l][y_r]
                D = integral[x_r][y_r]

                averaging[i][j] = (D + A - B - C) / ((x_r - x_l) * (y_r - y_l))

   ###########################
   # 9.  Сделать обратное преобразование YUV->BGR. Отобразить BGR.
   ###########################

    merged = cv2.merge([averaging, u, v])
    result = cv2.cvtColor(merged, cv2.COLOR_YUV2BGR)

    cv2.imshow('img', img)
    cv2.imshow('y', y)
    cv2.imshow("equalizeHist", eq_img)
    cv2.imshow("img_canny", img_canny)
    cv2.imshow("img_edges", corner_image)
    cv2.imshow("distance_image", distance_image)
    cv2.imshow('averaging', averaging)
    cv2.imshow('result', result)
    cv2.waitKey()