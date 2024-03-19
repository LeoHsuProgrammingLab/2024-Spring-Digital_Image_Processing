import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_histogram(hist, i):
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig('hist{}.png'.format(i))
    plt.close()

def show_histogram(hist):
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def equalize_img_by_normalize_hist(img, hist, scale_min = 0, scale_max = 255):
    total_pixels = img.shape[0] * img.shape[1]
    scale = scale_max - scale_min
    assert(hist.sum() == total_pixels)
    cdf = [0] * len(hist)
    
    cdf[0] = int(hist[0][0])
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + int(hist[i][0])

    normalized_mapping = [int((val / hist.sum()) * scale) for val in cdf]
    
    equalized_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            intensity = int(img[i, j]) - scale_min
            equalized_img[i, j] = normalized_mapping[intensity]
    
    return equalized_img

def equalize_img_locally(img, kernel_size = (5, 5)):
    equalized_img = img.copy()
    height, width = img.shape[:2]
    kernel_height, kernel_width = kernel_size
    for i in range(0, height, kernel_height):
        for j in range(0, width, kernel_width):
            tile = img[i:i+kernel_height, j:j+kernel_width]
            minimum = np.min(tile)
            maximum = np.max(tile)
            hist = cv2.calcHist([tile], [0], None, [maximum-minimum+1], [int(minimum), int(maximum+1)])
            equalized_kernal = minimum + equalize_img_by_normalize_hist(tile, hist, minimum, maximum+1)
            equalized_img[i:i+kernel_height, j:j+kernel_width] = equalized_kernal

    return equalized_img

def enhance_img(img):
    enhanced_img = inverse_function(img, 0.95)
    enhanced_img = square_root_img(enhanced_img)

    return enhanced_img

def square_root_img(img):
    square_root_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            intensity = int(img[i, j])
            square_root_img[i, j] = 255 * (intensity/255) ** (1/2)
            square_root_img[i, j] = 255 - square_root_img[i, j]

    return square_root_img

def inverse_function(img, threshold = 0.9):
    inverse_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            intensity = int(img[i, j])

            if intensity <= 255 * (1-threshold): # dark ones
                inverse_img[i, j] = ((255-intensity)/255 - threshold) / (1-threshold) * 255
            else: # white ones
                inverse_img[i, j] = 0

    return inverse_img