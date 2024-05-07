import cv2
import numpy as np
from P2 import *

def fourier_transform(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def inverse_fourier_transform(img):
    f_ishift = np.fft.ifftshift(img)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def apply_low_pass_filter(fshift):
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    r = 70  # radius of the circle
    center = [crow, ccol]  # center of the mask
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 1
    fshift_filtered = fshift * mask
    return fshift_filtered

def apply_gaussian_filter(fshift, cutoff_frequency):
    rows, cols = fshift.shape
    center_row, center_col = rows // 2, cols // 2
    x = np.linspace(0, rows, rows)
    y = np.linspace(0, cols, cols)
    x, y = np.meshgrid(y, x)
    radius = np.sqrt((x - center_row)**2 + (y - center_col)**2)
    gaussian = np.exp(-(radius**2) / (2 * (cutoff_frequency**2)))
    return fshift * gaussian

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

def p3():
    img = cv2.imread('hw4_sample_images/sample5.png', cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    # fourier transform
    fshift = fourier_transform(img)
    filtered_fshift = apply_low_pass_filter(fshift)
    # filtered_fshift = apply_gaussian_filter(fshift, 40)

    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    normalized_magnitude_spectrum = normalize(magnitude_spectrum)
    # normalized_magnitude_spectrum = median_filter(normalized_magnitude_spectrum, 3)

    magnitude_spectrum_ = 20 * np.log(np.abs(filtered_fshift) + 1)
    normalized_magnitude_spectrum_ = normalize(magnitude_spectrum_)
    # normalized_magnitude_spectrum_ = median_filter(normalized_magnitude_spectrum_, 3)

    filtered_img = inverse_fourier_transform(filtered_fshift)
    
    # cv2.imwrite('result5_1.png', normalized_magnitude_spectrum)
    # cv2.imwrite('result5_2.png', normalized_magnitude_spectrum_)
    
    cv2.imwrite('result5.png', filtered_img)

if __name__ == '__main__':
    p3()