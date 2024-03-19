import cv2
import numpy as np

def psnr(original_img, new_img):
    mse = np.mean((original_img - new_img) ** 2)
    if mse == 0:
        print('the images are the same')
        return 0
    
    return round(10 * np.log10(255**2 / mse), 2)

# low pass filter
def no_salt_filter(img):
    return pseudo_median_filter(img, type = 'maxmin')
    
# high pass filter
def no_pepper_filter(img):
    return pseudo_median_filter(img, type = 'minmax')

# median filter
def median_filter(img, kernel_size = 3):
    padding = kernel_size // 2

    padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    result_img = img.copy()

    for i in range(padding, img.shape[0] + padding):
        for j in range(padding, img.shape[1] + padding):
            kernel = padded_img[i-padding:i+padding+1, j-padding:j+padding+1]
            result_img[i-padding, j-padding] = np.median(kernel)
    
    return result_img

def cross_median_filter(img, cross_length = 5):
    padding = cross_length // 2

    padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    result_img = img.copy()

    for i in range(padding, img.shape[0] + padding):
        for j in range(padding, img.shape[1] + padding):
            horizontal = padded_img[i, j-padding:j+padding+1]
            vertical = padded_img[i-padding:i+padding+1, j]
            kernel = np.concatenate((horizontal, vertical))
            result_img[i-padding, j-padding] = np.median(kernel)
    
    return result_img

def pseudo_median_filter(img, cross_length = 7, kernel_size = 3, type = 'maxmin'):
    padding = cross_length // 2

    padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    result_img = img.copy()

    for i in range(padding, img.shape[0] + padding):
        for j in range(padding, img.shape[1] + padding):
            horizontal = padded_img[i, j-padding:j+padding+1]
            vertical = padded_img[i-padding:i+padding+1, j]
            if (type == 'maxmin'):
                horizontal_max_min = 0
                vertical_max_min = 0

                for k in range(cross_length-kernel_size+1):
                    min_val = np.min(horizontal[k:k+kernel_size])
                    if min_val > horizontal_max_min:
                        horizontal_max_min = min_val
                    min_val = np.min(vertical[k:k+kernel_size])
                    if min_val > vertical_max_min:
                        vertical_max_min = min_val
                result_img[i-padding, j-padding] = np.max([horizontal_max_min, vertical_max_min])
            elif (type == 'minmax'):
                horizontal_min_max = 255
                vertical_min_max = 255

                for k in range(cross_length-kernel_size+1):
                    max_val = np.max(horizontal[k:k+kernel_size])
                    if max_val < horizontal_min_max:
                        horizontal_min_max = max_val
                    max_val = np.max(vertical[k:k+kernel_size])
                    if max_val < vertical_min_max:
                        vertical_min_max = max_val
                result_img[i-padding, j-padding] = np.min([horizontal_min_max, vertical_min_max])
            else: # remove both
                horizontal_max_min = 0
                horizontal_min_max = 255
                vertical_max_min = 0
                vertical_min_max = 255
                for k in range(cross_length-kernel_size+1):
                    min_val = np.min(horizontal[k:k+kernel_size])
                    if min_val > horizontal_max_min:
                        horizontal_max_min = min_val
                    min_val = np.min(vertical[k:k+kernel_size])
                    if min_val > vertical_max_min:
                        vertical_max_min = min_val
                    max_val = np.max(horizontal[k:k+kernel_size])
                    if max_val < horizontal_min_max:
                        horizontal_min_max = max_val
                    max_val = np.max(vertical[k:k+kernel_size])
                    if max_val < vertical_min_max:
                        vertical_min_max = max_val
                        
                result_img[i-padding, j-padding] = 0.5*max(horizontal_max_min, vertical_max_min) + 0.5*min(horizontal_min_max, vertical_min_max)

    return result_img

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2)), (size, size))
    kernel /= np.sum(kernel)
    return kernel

def convolve2d(img, kernel):
    # print(img.shape, kernel.shape)
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2
    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    output = np.zeros_like(img)  # Create a blank image to hold the output
    if len(img.shape) == 2:
        image_padded = np.pad(img, ((padding, padding), (padding, padding)), mode='constant')
    else:
        image_padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

    height, width = img.shape[:2]
    channel = 1 if len(img.shape) == 2 else img.shape[2]
    if channel == 1:
        for x in range(height):
            for y in range(width):
                output[x, y] = np.sum(image_padded[x:x+kernel_size, y:y+kernel_size] * kernel)
    else:
        for c in range(channel):
            for x in range(height):
                for y in range(width):
                    output[x, y] = np.sum(image_padded[x:x+kernel_size, y:y+kernel_size, c] * kernel)
            
    return output

def gaussian_filter(img, size, sigma): # Gaussian Blur
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(img, kernel)
