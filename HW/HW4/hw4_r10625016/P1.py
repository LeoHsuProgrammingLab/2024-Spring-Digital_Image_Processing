import cv2
import numpy as np

# Digital halftoning

def expand_mtx(mtx):
    new_mtx = np.zeros((2 * len(mtx), 2 * len(mtx)))
    top_left_bock = 4 * mtx + 1
    top_right_block = 4 * mtx + 2
    bottom_left_block = 4 * mtx + 3
    bottom_right_block = 4 * mtx

    new_mtx[:len(mtx), :len(mtx)] = top_left_bock
    new_mtx[:len(mtx), len(mtx):] = top_right_block
    new_mtx[len(mtx):, :len(mtx)] = bottom_left_block
    new_mtx[len(mtx):, len(mtx):] = bottom_right_block
    
    return new_mtx

def dithering(img, dithering_matrix, size):
    mtx_size = len(dithering_matrix)
    dithering_mtx = dithering_matrix

    while (mtx_size < size):
        dithering_mtx = expand_mtx(dithering_mtx)
        mtx_size *= 2
    threshold_matrix = (dithering_mtx + 0.5) * 255 / (mtx_size ** 2)

    height, width = img.shape
    dithered_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            dithered_img[i, j] = 255 if img[i, j] > threshold_matrix[i % mtx_size, j % mtx_size] else 0

    return dithered_img

def p1_a():
    img = cv2.imread('hw4_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    dithering_matrix = np.array([[1, 2], [3, 0]])
    dithered_img = dithering(img, dithering_matrix, 2)
    cv2.imwrite('result1.png', dithered_img)

def p1_b():
    img = cv2.imread('hw4_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    dithering_matrix = np.array([[1, 2], [3, 0]])
    # dithering_matrix = np.array([[1, 3], [0, 2]])

    dithered_img = dithering(img, dithering_matrix, 256)
    cv2.imwrite('result2.png', dithered_img)

# Error diffusion
def folyd_steinberg_dithering(img):
    height, width = img.shape
    dithered_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            old_pixel = img[i, j]
            new_pixel = 255 if old_pixel > 127 else 0
            dithered_img[i, j] = new_pixel
            error = old_pixel - new_pixel
            if j + 1 < width:
                img[i, j + 1] += error * 7 / 16
            if i + 1 < height:
                img[i + 1, j] += error * 5 / 16
                if j - 1 >= 0:
                    img[i + 1, j - 1] += error * 3 / 16
                if j + 1 < width:
                    img[i + 1, j + 1] += error * 1 / 16

    return dithered_img

def jarvis_judice_ninke_dithering(img):
    height, width = img.shape
    dithered_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            old_pixel = img[i, j]
            new_pixel = 255 if old_pixel > 127 else 0
            dithered_img[i, j] = new_pixel
            error = old_pixel - new_pixel
            if j + 1 < width:
                img[i, j + 1] += error * 7 / 48
                if j + 2 < width:
                    img[i, j + 2] += error * 5 / 48
            if i + 1 < height:
                if j - 2 >= 0:
                    img[i + 1, j - 2] += error * 3 / 48
                if j - 1 >= 0:
                    img[i + 1, j - 1] += error * 5 / 48
                img[i + 1, j] += error * 7 / 48
                if j + 1 < width:
                    img[i + 1, j + 1] += error * 5 / 48
                if j + 2 < width:
                    img[i + 1, j + 2] += error * 3 / 48
            if i + 2 < height:
                if j - 2 >= 0:
                    img[i + 2, j - 2] += error * 1 / 48
                if j - 1 >= 0:
                    img[i + 2, j - 1] += error * 3 / 48
                img[i + 2, j] += error * 5 / 48
                if j + 1 < width:
                    img[i + 2, j + 1] += error * 3 / 48
                if j + 2 < width:
                    img[i + 2, j + 2] += error * 1 / 48

    return dithered_img

def sierra_dithering(img):
    height, width = img.shape
    dithered_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            old_pixel = img[i, j]
            new_pixel = 255 if old_pixel > 127 else 0
            dithered_img[i, j] = new_pixel
            error = old_pixel - new_pixel
            if j + 1 < width:
                img[i, j + 1] += error * 5 / 32
                if j + 2 < width:
                    img[i, j + 2] += error * 3 / 32
            if i + 1 < height:
                if j - 2 >= 0:
                    img[i + 1, j - 2] += error * 2 / 32
                if j - 1 >= 0:
                    img[i + 1, j - 1] += error * 4 / 32
                img[i + 1, j] += error * 5 / 32
                if j + 1 < width:
                    img[i + 1, j + 1] += error * 4 / 32
                if j + 2 < width:
                    img[i + 1, j + 2] += error * 2 / 32
            if i + 2 < height:
                if j - 2 >= 0:
                    img[i + 2, j - 2] += error * 2 / 32
                if j - 1 >= 0:
                    img[i + 2, j - 1] += error * 3 / 32
                img[i + 2, j] += error * 4 / 32
                if j + 1 < width:
                    img[i + 2, j + 1] += error * 3 / 32
                if j + 2 < width:
                    img[i + 2, j + 2] += error * 2 / 32

    return dithered_img

def stucki_dithering(img):
    height, width = img.shape
    dithered_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            old_pixel = img[i, j]
            new_pixel = 255 if old_pixel > 127 else 0
            dithered_img[i, j] = new_pixel
            error = old_pixel - new_pixel
            if j + 1 < width:
                img[i, j + 1] += error * 8 / 42
                if j + 2 < width:
                    img[i, j + 2] += error * 4 / 42
            if i + 1 < height:
                if j - 2 >= 0:
                    img[i + 1, j - 2] += error * 2 / 42
                if j - 1 >= 0:
                    img[i + 1, j - 1] += error * 4 / 42
                img[i + 1, j] += error * 8 / 42
                if j + 1 < width:
                    img[i + 1, j + 1] += error * 4 / 42
                if j + 2 < width:
                    img[i + 1, j + 2] += error * 2 / 42
            if i + 2 < height:
                if j - 2 >= 0:
                    img[i + 2, j - 2] += error * 1 / 42
                if j - 1 >= 0:
                    img[i + 2, j - 1] += error * 2 / 42
                img[i + 2, j] += error * 4 / 42
                if j + 1 < width:
                    img[i + 2, j + 1] += error * 2 / 42
                if j + 2 < width:
                    img[i + 2, j + 2] += error * 1 / 42

    return dithered_img

def p1_c():
    img = cv2.imread('hw4_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    folyd_steinberg_dithered_img = folyd_steinberg_dithering(img)
    cv2.imwrite('result3.png', folyd_steinberg_dithered_img)
    jarvis_judice_ninke_dithered_img = jarvis_judice_ninke_dithering(img)
    cv2.imwrite('result4.png', jarvis_judice_ninke_dithered_img)
    # sierra_dithered_img = sierra_dithering(img)
    # cv2.imwrite('result4_1.png', sierra_dithered_img)
    # stucki_dithered_img = stucki_dithering(img)
    # cv2.imwrite('result4_2.png', stucki_dithered_img)

if __name__ == '__main__':
    p1_a()
    p1_b()
    p1_c()