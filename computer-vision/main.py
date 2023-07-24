import cv2 as cv


def binary_thresholding(t):
    img = cv.imread('source_images/digits.png', 0)
    ret, thresh = cv.threshold(img, t, 255, cv.THRESH_BINARY)
    cv.imwrite("final_images/binary_digits_" + str(t) + ".png", thresh);


def inverse_binary_thresholding(t):
    img = cv.imread('source_images/digits.png', 0)
    ret, thresh = cv.threshold(img, t, 255, cv.THRESH_BINARY_INV)
    cv.imwrite("final_images/inverse_binary_digits_" + str(t) + ".png", thresh);


def window_binary_thresholding(t1, t2):
    img = cv.imread('source_images/digits.png', 0)
    thresh = cv.inRange(img, t1, t2)
    cv.imwrite("final_images/window_binary_digits_" + str(t2) + ".png", thresh)


def canny_edge_detection():
    kernel = [3, 7, 11]
    low = [50, 100, 150]
    high = [100, 150, 200]

    for k in kernel:
        for l in low:
            for h in high:
                img = cv.imread('source_images/dolphin.png', cv.IMREAD_GRAYSCALE)
                blur = cv.GaussianBlur(img, (k, k), 0)
                edge = cv.Canny(blur, l, h)
                cv.imwrite('final_images/canny_edge_detection_' + str(k) + str(l) + str(h) + '.png', edge)


def average_filter_clean():
    kernel_gaus = [3, 5, 7, 9]
    for k in kernel_gaus:
        img = cv.imread('source_images/clean.png', cv.IMREAD_GRAYSCALE)
        # aver2 = cv.blur(img, (k, k))
        aver = cv.boxFilter(img, -1, (k, k), normalize=True)
        cv.imwrite("final_images/averaging_filter_clean_" + str(k) + ".png", aver)


def average_filter_noisy():
    kernel_gaus = [3, 5, 7, 9]
    for k in kernel_gaus:
        img = cv.imread('source_images/noisy.png', cv.IMREAD_GRAYSCALE)
        # aver2 = cv.blur(img, (k, k))
        aver = cv.boxFilter(img, -1, (k, k), normalize=True)
        cv.imwrite("final_images/averaging_filter_noisy_" + str(k) + ".png", aver)


def gaussian_filter_clean():
    sigma = [0.5, 1, 2, 4]
    for s in sigma:
        img = cv.imread('source_images/clean.png', cv.IMREAD_GRAYSCALE)
        blur = cv.GaussianBlur(img, (9, 9), s)
        cv.imwrite("final_images/gaussian_filter_clean_9" + str(s) + ".png", blur)


def gaussian_filter_noisy():
    sigma = [0.5, 1, 2, 4]
    for s in sigma:
        img = cv.imread('source_images/noisy.png', cv.IMREAD_GRAYSCALE)
        blur = cv.GaussianBlur(img, (9, 9), s)
        cv.imwrite("final_images/gaussian_filter_noisy_9" + str(s) + ".png", blur)


if __name__ == "__main__":
    thresh = [55, 129, 200, 130]
    for t in thresh:
        binary_thresholding(t)
        inverse_binary_thresholding(t)

    window_binary_thresholding(55, 90)
    window_binary_thresholding(90, 129)
    window_binary_thresholding(90, 130)
    window_binary_thresholding(129, 200)

    average_filter_noisy()
    gaussian_filter_noisy()
    canny_edge_detection()
