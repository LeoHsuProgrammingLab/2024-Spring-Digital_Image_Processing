import cv2
from filters import *

# (a): pepper_filter and combine filter
def p2_a(sample5, sample6):
    s5 = cv2.imread(sample5, cv2.IMREAD_GRAYSCALE)
    s6 = cv2.imread(sample6, cv2.IMREAD_GRAYSCALE)

    gaussian_img = gaussian_filter(s5, size = 3, sigma = 1)
    # for i in range(5):
    # median_img_s5 = median_filter(s5)
    pseudo_median_img = pseudo_median_filter(gaussian_img, cross_length = 7, kernel_size = 3, type = 'maxmin')
    pseudo_median_img = pseudo_median_filter(pseudo_median_img, cross_length = 7, kernel_size = 3, type = 'minmax')
    pseudo_median_img = pseudo_median_filter(pseudo_median_img, cross_length = 7, kernel_size = 3, type = 'both')
    """
    for size = 3, sigma is between 0.5 to 1
    for size = 5, sigma is between 1 to 1.5
    for size = 7, sigma is between 1.5 to 2
    """
    median_img = median_filter(s6)

    cv2.imwrite('result10.png', pseudo_median_img)
    cv2.imwrite('result11.png', median_img)


# (b): psnr
def p2_b(original_img, recovered_img1, recovered_img2):
    origin = cv2.imread(original_img, cv2.IMREAD_GRAYSCALE)
    recovered1 = cv2.imread(recovered_img1, cv2.IMREAD_GRAYSCALE)
    recovered2 = cv2.imread(recovered_img2, cv2.IMREAD_GRAYSCALE)
    print(f"1st PSNR: {round(psnr(origin, recovered1), 2)}")
    print(f"2nd PSNR: {round(psnr(origin, recovered2), 2)}")

# (c): bonus noise_remove
def p2_c(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # noise_remove_img = median_filter(img) # psnr: 30.73
    gaussian_img = gaussian_filter(img, size = 3, sigma = 1) # psnr: 30.73

    pseudo_median_img = pseudo_median_filter(gaussian_img, cross_length = 7, kernel_size = 3, type = 'maxmin')
    pseudo_median_img = pseudo_median_filter(pseudo_median_img, cross_length = 7, kernel_size = 3, type = 'minmax')
    noise_remove_img = pseudo_median_filter(pseudo_median_img, cross_length = 7, kernel_size = 3, type = 'both') # psnr: 30.75
    noise_remove_img = median_filter(noise_remove_img)
    cv2.imwrite('result12.png', noise_remove_img)

if __name__ == "__main__":
    p2_a('hw1_sample_images/sample5.png', 'hw1_sample_images/sample6.png')
    p2_b('hw1_sample_images/sample4.png', 'result10.png', 'result11.png')
    p2_c('hw1_sample_images/sample7.png')
    # print(f"result12.png PSNR: {psnr(cv2.imread('hw1_sample_images/sample4.png'), cv2.imread('result12.png'))}")
    