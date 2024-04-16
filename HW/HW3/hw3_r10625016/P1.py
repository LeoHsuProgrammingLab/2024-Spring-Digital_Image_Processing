import cv2
import numpy as np
import matplotlib.pyplot as plt
import filters

def erode(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=(0, 0))
    
    eroded_img = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            patch = padded_img[i:i+kh, j:j+kw]
            eroded_img[i, j] = np.min(patch[kernel == 1])
    return eroded_img

def morphological_boundary(img):
    kernel = np.ones((3,3))
    # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    erosion = erode(img, kernel)
    boundary = img - erosion
    return boundary

def p1_a():
    img = cv2.imread('hw3_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    # erosion then subtract from original image
    boundary = morphological_boundary(img)
    cv2.imwrite('result1.png', boundary)

def dilate(img, kernel):    
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=(0, 0))
    
    dilated_img = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            patch = padded_img[i:i+kh, j:j+kw]
            dilated_img[i, j] = np.max(patch[kernel == 1])
    return dilated_img

def hole_filling(img, kernel, iterations=7):
    '''
    Steps:
    1. Get F'
    2. Dilate 
    '''
    # hist = cv2.calcHist([img], [0], None, [256], [0,256])
    # plt.plot(hist)
    # plt.show()
    f = img.copy()
    fc = 255 - img
    for i in range(iterations):
        f = dilate(f, kernel)
        f = cv2.bitwise_and(f, fc) # intersection
    
    result = cv2.bitwise_or(f, img)
    return result

def p1_b():
    img = cv2.imread('hw3_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    # hole filling
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    result = hole_filling(img, kernel, iterations=7)
    cv2.imwrite('result2.png', result)

def opening_noise_removal(img):
    # opening operation
    kernel = np.ones((3,3))
    img_ = erode(img, kernel)
    cleaned_img = dilate(img_, kernel)
    return cleaned_img

def closing_noise_removal(img):
    # closing operation
    kernel = np.ones((3,3))
    img_ = dilate(img, kernel)
    cleaned_img = erode(img_, kernel)
    return cleaned_img

def p1_c():
    img = cv2.imread('hw3_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    
    # noise removal using morphological operations
    cleaned_img = opening_noise_removal(img)
    cv2.imwrite('result3.png', cleaned_img)

    # close_img = closing_noise_removal(img)
    # cv2.imwrite('result3_2.png', close_img)

    # pepper_removal = filters.no_pepper_filter(img)
    # cv2.imwrite('result3_3.png', pepper_removal)

    # # noise removal using median filter
    # median_img = filters.median_filter(img)
    # cv2.imwrite('result3_1.png', median_img)

def connected_component_labeling(img, iterations=7):
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/label.htm
    kernel = np.ones((3,3))
    f = img.copy()

    for i in range(iterations):
        dilated_img = dilate(f, kernel)
        f = cv2.bitwise_and(dilated_img, img)
    
    cv2.imwrite('result_1d.png', f)
    return f

def connected_component_labeling(binary_img):
    label_matrix = np.zeros_like(binary_img)
    label_equivalences = {}
    label_counter = 1
    
    def find_equivalent_label(label):
        while label in label_equivalences:
            label = label_equivalences[label]
        return label
    
    for y in range(binary_img.shape[0]):
        for x in range(binary_img.shape[1]):
            if binary_img[y, x] == 255:
                neighbors = [label_matrix[y-1, x], label_matrix[y, x-1], label_matrix[y-1, x-1], label_matrix[y-1, x+1]]
                foreground_neighbors = [neighbor for neighbor in neighbors if neighbor != 0]
                
                if len(foreground_neighbors) == 0:
                    label_matrix[y, x] = label_counter
                    label_counter += 1
                elif len(foreground_neighbors) == 1:
                    label_matrix[y, x] = find_equivalent_label(foreground_neighbors[0])
                else:
                    min_label = min(foreground_neighbors)
                    label_matrix[y, x] = min_label
                    for neighbor in foreground_neighbors:
                        if neighbor != min_label:
                            label_equivalences[neighbor] = min_label
    
    for y in range(binary_img.shape[0]):
        for x in range(binary_img.shape[1]):
            label_matrix[y, x] = find_equivalent_label(label_matrix[y, x])
    
    return label_matrix

def draw_different_labels(img, label_matrix):
    unique_labels = np.unique(label_matrix)
    colors = np.random.randint(0, 255, (len(unique_labels), 3))
    result = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if label_matrix[y, x] != 0:
                result[y, x] = colors[np.where(unique_labels == label_matrix[y, x])]
    
    return result

def p1_d():
    img = cv2.imread('hw3_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
    # noise removal using morphological operations
    cleaned_img = opening_noise_removal(img)

    # cleaned_img = erode(cleaned_img, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    cleaned_img = hole_filling(cleaned_img, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), iterations=1)
    
    # connected component labeling
    labeled_img = connected_component_labeling(cleaned_img)

    unique_labels = np.unique(labeled_img)
    print(unique_labels)
    print('Number of connected components:', len(unique_labels) - 1)
    labeled_img = draw_different_labels(img, labeled_img)

    # cv2.imwrite('result_3d.png', labeled_img)
    
if __name__ == "__main__":
    p1_a()
    p1_b()
    p1_c()
    p1_d()