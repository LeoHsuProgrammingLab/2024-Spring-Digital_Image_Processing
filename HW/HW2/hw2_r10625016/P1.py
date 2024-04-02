import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Utils
def sobel_alg(img, weight = 1):
    # img: input image (grayscale)
    img_h, img_w = img.shape
        
    kernel_r = np.array([
        [-1, 0, 1], 
        [-2, 0, 2], 
        [-1, 0, 1]
    ], dtype=np.float32)
    
    kernel_c = np.array([
        [1, 2, 1], 
        [0, 0, 0], 
        [-1, -2, -1]
    ], dtype=np.float32)

    gradient_r = (convolve2d(img, kernel_r) * weight).astype(np.int32) 
    gradient_c = (convolve2d(img, kernel_c) * weight).astype(np.int32) 

    magnitude = np.sqrt(gradient_r**2 + gradient_c**2)
    theta = np.arctan2(gradient_c, gradient_r) * 180 / np.pi

    return magnitude, theta

def convolve2d(img, kernel):
    # img: input image (grayscale)
    # kernel: kernel for convolution
    # return: convoluted image
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant')
    
    output_img = np.zeros_like(img)

    for i in range(img_h):
        for j in range(img_w):
            tile = padded_img[i:i+kernel_h, j:j+kernel_w]
            output_img[i, j] = np.sum(tile * kernel)
    
    return output_img

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2)), (size, size))
    kernel /= np.sum(kernel)
    return kernel

def gaussian_filter(img, size, sigma): # Gaussian Blur
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(img, kernel)

def non_max_suppression(magnitude, theta): # theta is the direction of gradient, already turn into angle
    img_h, img_w = magnitude.shape
    output_img = np.zeros_like(magnitude)
    theta[theta < 0] += 180

    for i in range(1, img_h-1):
        for j in range(1, img_w-1):
            angle = theta[i, j]

            # angle = 0
            if (0 <= angle < 22.5) or (180 >= angle >= 157.5):
                prev_pixel = magnitude[i, j-1]
                next_pixel = magnitude[i, j+1]
            # angle = 45
            elif (angle >= 22.5 and angle < 67.5):
                prev_pixel = magnitude[i-1, j+1]
                next_pixel = magnitude[i+1, j-1]
            # angle = 90
            elif (angle >= 67.5 and angle < 112.5):
                prev_pixel = magnitude[i-1, j]
                next_pixel = magnitude[i+1, j]
            # angle = 135
            elif (angle >= 112.5 and angle < 157.5):
                prev_pixel = magnitude[i-1, j-1]
                next_pixel = magnitude[i+1, j+1]

            if magnitude[i, j] >= prev_pixel and magnitude[i, j] >= next_pixel:
                output_img[i, j] = magnitude[i, j]
    
    return output_img

def double_thresholding(img, higher_bound = 0.75, lower_bound = 0.45):
    img_h, img_w = img.shape
    output_img = np.zeros_like(img)

    higher_threshold = np.max(img) * higher_bound
    lower_threshold = np.max(img) * lower_bound

    weak = 127
    strong = 255
    # zero = 0

    strong_i, strong_j = np.where(img >= higher_threshold)
    weak_i, weak_j = np.where((img >= lower_threshold) & (img < higher_threshold))

    output_img[strong_i, strong_j] = strong
    output_img[weak_i, weak_j] = weak   

    return output_img

def connected_component_labeling(img):
    img_h, img_w = img.shape
    output_img = np.zeros_like(img)

    for i in range(img_h):
        for j in range(img_w):
            if img[i, j] == 127:
                connected_component_labeling_recursive(img, i, j)
            elif img[i, j] == 255:
                output_img[i, j] = 255
    
    return output_img

def connected_component_labeling_recursive(img, i, j):
    img_h, img_w = img.shape

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    for n in neighbors:
        if i + n[0] < 0 or i + n[0] >= img_h or j + n[1] < 0 or j + n[1] >= img_w:
            continue
        if img[i+n[0], j+n[1]] == 127:
            img[i, j] = connected_component_labeling_recursive(img, i+n[0], j+n[1])
        if img[i+n[0], j+n[1]] == 255:
            return 255
        else:
            return 0

def enhanced_noise_reduction(img):
    gaussian_filtered_img = gaussian_filter(img, 3, 1) # sigma = 1, kernel size = 3
    gaussian_filtered_img = gaussian_filter(gaussian_filtered_img, 5, 1.5)
    gaussian_filtered_img = gaussian_filter(gaussian_filtered_img, 7, 2)

    return gaussian_filtered_img

def noise_reduction(img):
    gaussian_filtered_img = gaussian_filter(img, 7, 2) # sigma = 2, kernel size = 7

    return gaussian_filtered_img

def canny_alg(img, enhanced = False):
    if enhanced:
        img = enhanced_noise_reduction(img)
    else:
        img = noise_reduction(img)
    magnitude, theta = sobel_alg(img)
    non_max_suppressed_img = non_max_suppression(magnitude, theta)
    thresholded_img = double_thresholding(non_max_suppressed_img)
    connected_component_img = connected_component_labeling(thresholded_img)

    return connected_component_img

def lapacian_of_gaussian_alg(img, weight = 1/4):
    gaussian_filtered_img = gaussian_filter(img, 3, 1) # sigma = 1, kernel size = 3     
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.int32)

    lapacian_img = convolve2d(gaussian_filtered_img, laplacian_kernel) # have to binarize
    lapacian_img = lapacian_img * weight

    return lapacian_img

def zero_crossing_detection(img, threshold=0.5):
    hist, bins= np.histogram(img, bins=100, range=(img.min(), img.max()))

    # draw histogram
    # plot_histogram(hist, 4, xlabel='Bins')

    # thresholding
    right_threshold = np.mean(img) + np.std(img) * threshold
    left_threshold = np.mean(img) - np.std(img) * threshold
    img[img > right_threshold] = 255
    img[img < left_threshold] = 255
    img[img != 255] = 0

    # zero crossing detection
    img_h, img_w = img.shape
    output_img = np.zeros_like(img)

    for i in range(1, img_h-1):
        for j in range(1, img_w-1):
            if img[i, j] == 0:
                neighbors = [img[i-1, j], img[i+1, j], img[i, j-1], img[i, j+1]]
                if any(np.sign(neighbors[k]) != np.sign(neighbors[k+1]) for k in range(len(neighbors)-1)):
                    # print(neighbors)
                    output_img[i, j] = 255

    return output_img
    
def plot_histogram(hist, i, title = 'Histogram', xlabel = 'Pixel Value', ylabel = 'Frequency'):
    plt.plot(hist)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('hist{}.png'.format(i))
    plt.close()

def edge_crispening(img, kernel_size = 3, c = 2/3):
    gaussian_filtered_img = gaussian_filter(img, kernel_size, 1) # sigma = 1, kernel size = 3
    # unsharp masking
    edge_crispening_img = img.copy().astype(np.float32)
    img_h, img_w = img.shape
    for i in range(img_h):
        for j in range(img_w):
            edge_crispening_img[i, j] = img[i, j] * c / (2 * c - 1) - gaussian_filtered_img[i, j] * (1 - c) / (2 * c - 1)
    
    return edge_crispening_img

def hough_transform(img, threshold=100):
    # https://docs.opencv.org/3.1.0/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm
    img_h, img_w = img.shape
    max_rho = int(math.sqrt(img_h**2 + img_w**2))
    accumulator = np.zeros((2*max_rho, 180), dtype=np.uint32)

    for i in range(img_h):
        for j in range(img_w):
            if img[i, j] == 255:
                for theta in range(180):
                    # rho = x * cos(theta) + y * sin(theta)
                    rho = int(i * np.cos(np.deg2rad(theta)) + j * np.sin(np.deg2rad(theta)))
                    accumulator[rho, theta] += 1

    lines = []
    for rho in range(accumulator.shape[0]):
        for theta in range(accumulator.shape[1]):
            if accumulator[rho, theta] > threshold:
                lines.append((rho, theta))

    return lines

def draw_lines(img, lines):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for rho, theta in lines:
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        x2 = int(x0 - 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imwrite('result7.png', img)

def draw_lines_(img, lines):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for line in lines:
        rho, theta = line[0]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        x2 = int(x0 - 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    cv2.imwrite('result7_2.png', img)

def thresholding(img, threshold):
    img[img < threshold] = 0
    img[img >= threshold] = 255
    return img

# Main
def p1_a():
    img = cv2.imread('hw2_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    magnitude, theta = sobel_alg(img)
    cv2.imwrite('result1.png', magnitude)
    
    # Thresholding
    threshold = np.mean(magnitude) + np.std(magnitude) * 0.5
    sobel_img = magnitude.copy()
    sobel_img = thresholding(sobel_img, threshold)
    cv2.imwrite('result2.png', sobel_img)

def p1_b():
    img = cv2.imread('hw2_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    canny_img = canny_alg(img, enhanced=True)
    cv2.imwrite('result3.png', canny_img)

def p1_c():
    img = cv2.imread('hw2_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    lapacian_img = lapacian_of_gaussian_alg(img, 1)
    result_img = zero_crossing_detection(lapacian_img, 0.9)
    cv2.imwrite('result4.png', result_img)

def p1_d():
    img = cv2.imread('hw2_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
    edge_crispening_img = edge_crispening(img, c = 3/5)
    # assert np.array_equal(edge_crispening_img, img)
    cv2.imwrite('result5.png', edge_crispening_img)

def p1_e():
    img_ = cv2.imread('result5.png', cv2.IMREAD_GRAYSCALE)
    img = img_.copy()
    for i in range(5):
        img = gaussian_filter(img, 5, 2) # sigma = 1, kernel size = 3
    canny_img = canny_alg(img)
    cv2.imwrite('result6.png', canny_img)

    # canny_img_2 = cv2.Canny(img, 50, 80)
    # cv2.imwrite('result6_2.png', canny_img_2)
    
    r6 = cv2.imread('result6.png', cv2.IMREAD_GRAYSCALE)
    hough_lines = hough_transform(r6, 80) 
    draw_lines(r6, hough_lines)

    # hough_lines_ = cv2.HoughLines(r6, 1, np.pi/180, 100)
    # draw_lines_(r6, hough_lines_)
    # cv2.imwrite('result7_2.png', r6)

if __name__ == '__main__':
    p1_a()
    p1_b()
    p1_c()
    p1_d()
    p1_e()