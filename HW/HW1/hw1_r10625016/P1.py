import cv2
from utils import *

def p1_a(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]

    for i in range(width):
        for j in range(height):
            img[j, i] = img[j, i] / 3
    
    cv2.imwrite('result3.png', img)

def p1_b(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]

    for i in range(width):
        for j in range(height):
            img[j, i] = img[j, i] * 3
    
    img = img.clip(0, 255)
    print(img.max(), img.min())
    cv2.imwrite('result4.png', img)

def p1_c(filename1, filename2, filename3):
    img1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(filename3, cv2.IMREAD_GRAYSCALE)

    for i, img in enumerate([img1, img2, img3]):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plot_histogram(hist, i)

def p1_d(filename1, filename2, filename3):
    img1 = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(filename3, cv2.IMREAD_GRAYSCALE)

    for i, img in enumerate([img1, img2, img3]):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        equalized_img = equalize_img_by_normalize_hist(img, hist)
        equalized_hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
        cv2.imwrite('result{}.png'.format(i+5), equalized_img)
        plot_histogram(equalized_hist, i+5)

def p1_e(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    equalized_img = equalize_img_locally(img)
    equalized_hist = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
    cv2.imwrite('result8.png', equalized_img)
    plot_histogram(equalized_hist, 8)

def p1_f(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    enhanced_img = enhance_img(img)
    enhanced_hist = cv2.calcHist([enhanced_img], [0], None, [256], [0, 256])
    plot_histogram(enhanced_hist, 9)
    cv2.imwrite('result9.png', enhanced_img)


if __name__ == "__main__":
    p1_a('hw1_sample_images/sample2.png')
    p1_b('result3.png')
    p1_c('hw1_sample_images/sample2.png', 'result3.png', 'result4.png')
    p1_d('hw1_sample_images/sample2.png', 'result3.png', 'result4.png')
    p1_e('hw1_sample_images/sample2.png')
    p1_f('hw1_sample_images/sample3.png')