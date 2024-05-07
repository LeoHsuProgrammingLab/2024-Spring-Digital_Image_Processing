import numpy as np
import cv2
from OCR_match import *
from utils import *

def segment_characters_cc(binary_image, min_size):
    # Use connected components to find letters
    num_labels, labels_im = cv2.connectedComponents(binary_image, connectivity=8)
    characters = []
    bounding_boxes = []

    for label in range(1, num_labels):
        component = labels_im == label
        if np.sum(component) > min_size:  # Filter out small components
            character = np.full_like(binary_image, 255)
            character[component] = 0

            # Calculate the horizontal and vertical positions where the component exists
            active_pixels = np.argwhere(component)
            x_min = np.min(active_pixels[:, 1])  # min column index
            y_min = np.min(active_pixels[:, 0])  # min row index
            x_max = np.max(active_pixels[:, 1])  # max column index
            y_max = np.max(active_pixels[:, 0])  # max row index

            # Append the character image and its bounding box starting x-coordinate
            bounding_boxes.append((x_min, character))

    # Sort characters based on the x coordinate of their bounding boxes
    bounding_boxes.sort(key=lambda b: b[0])
    sorted_characters = [b[1] for b in bounding_boxes]
    # for char in sorted_characters:
    #     show_img(char)

    return sorted_characters

def segment_characters(binary_image, transect_line_len):
    h, w = binary_image.shape
    horizontal_cut_points = vertical_transect_line(binary_image, transect_line_len, w)
    horizontal_cut_points = [0] + horizontal_cut_points[1:]
    horizontal_cut_points = [0 if i < 0 else i for i in horizontal_cut_points]
    
    character_images = []
    for i in range(0, len(horizontal_cut_points)-1, 2):
        character_images.append(binary_image[:, horizontal_cut_points[i]:horizontal_cut_points[i+1]])
    # for i in character_images:
    #     show_img(i)
    
    return character_images

def recognize_characters(character_images, templates):
    recognized_text = ''
    for char_img in character_images:
        char_img = np.pad(char_img, (5, 5), 'constant', constant_values=255)
        # print(np.unique(char_img))
        match = match_template(char_img, templates)
        if match:
            recognized_text += match
    return recognized_text

def generate_templates(img):
    templates = {}
    vertical_points = horizontal_transect_line(img)
    vertical_points = [0] + vertical_points + [img.shape[0]]

    cut_points = []
    for i in range(0, len(vertical_points)-1, 2):
        cut_points.append(int((vertical_points[i] + vertical_points[i+1]) / 2))

    three_parts = []
    for i in range(len(cut_points)-1):
        cut_img = img[cut_points[i]:cut_points[i+1], :]
        h, w = cut_img.shape
        horizontal_points = vertical_transect_line(cut_img, h, w)
        horizontal_points = [0] + horizontal_points + [cut_img.shape[1]]
        cutpoints = []
        for j in range(0, len(horizontal_points)-1, 2):
            cutpoints.append(int((horizontal_points[j] + horizontal_points[j+1]) / 2))
        three_parts.append(cutpoints)

    for i in range(len(three_parts[0])-1): # A ~ L : 12
        target = img[cut_points[0]:cut_points[1], three_parts[0][i]:three_parts[0][i+1]]
        target = custom_threshold(target, 128, 255)
        templates[chr(i+65)] = target

    for i in range(len(three_parts[1])-1): # M ~ W : 11
        target = img[cut_points[1]:cut_points[2], three_parts[1][i]:three_parts[1][i+1]]
        target = custom_threshold(target, 128, 255)
        templates[chr(i+77)] = target

    for i in range(len(three_parts[2])-1): # X ~ Z + 0 ~ 9 : 13
        if i+88 < 91:
            target = img[cut_points[2]:cut_points[3], three_parts[2][i]:three_parts[2][i+1]]
            target = custom_threshold(target, 128, 255)
            templates[chr(i+88)] = target
        else:
            target = img[cut_points[2]:cut_points[3], three_parts[2][i]:three_parts[2][i+1]]
            target = custom_threshold(target, 128, 255)
            templates[chr(i+45)] = target

    return templates

def horizontal_transect_line(img):
    height, width = img.shape
    vertical_points = []
    for i in range(height):
        if np.any(img[i, :] == 0) and np.all(img[i-1, :] == 255):
            vertical_points.append(i-1)   
        if np.any(img[i, :] == 0) and np.all(img[i+1, :] == 255):
            vertical_points.append(i+1)
    
    return vertical_points

def vertical_transect_line(img, h, width):
    horizontal_points = []
    for i in range(width):
        if np.any(img[:int(h), i] == 0) and np.all(img[:int(h), i-1] == 255):
            horizontal_points.append(i-1)
        if i < width - 1 and np.any(img[:int(h), i] == 0) and np.all(img[:int(h), i+1] == 255):
            horizontal_points.append(i+1)
    
    return horizontal_points


def custom_threshold(image, thresh_value, max_value):
    # Create a copy of the image to avoid modifying the original
    binary_image = np.copy(image)
    
    # Apply the threshold: Set pixels above the threshold to max_value, others to 0
    binary_image[binary_image > thresh_value] = max_value
    binary_image[binary_image <= thresh_value] = 0
    
    return binary_image

def median_filter(image, kernel_size):
    if len(image.shape) == 2:
        h, w = image.shape
        filtered_img = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                if i - kernel_size >= 0 and i + kernel_size < h and j - kernel_size >= 0 and j + kernel_size < w:
                    filtered_img[i, j] = np.median(image[i-kernel_size:i+kernel_size, j-kernel_size:j+kernel_size])
                else:
                    filtered_img[i, j] = image[i, j]
    else:
        h, w, c= image.shape
        filtered_img = np.zeros((h, w, c), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    if i - kernel_size >= 0 and i + kernel_size < h and j - kernel_size >= 0 and j + kernel_size < w:
                        filtered_img[i, j, k] = np.median(image[i-kernel_size:i+kernel_size, j-kernel_size:j+kernel_size, k])
                    else:
                        filtered_img[i, j, k] = image[i, j, k]

    return filtered_img

def sample2_preprocessing(sample_path):
    sample2 = cv2.imread(sample_path)
    sample2_gray = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    
    h, w, c = sample2.shape

    red_c = sample2[:, :, 2]
    green_c = sample2[:, :, 1]
    blue_c = sample2[:, :, 0]

    red_dominate = (red_c > 200) & (green_c < 180) & (blue_c < 180)

    mask = np.uint8(red_dominate) * 255
    indices = np.where(mask == 255)

    top = np.min(indices[0]) - 1
    bottom = np.max(indices[0]) + 1
    left = np.min(indices[1]) - 1
    right = np.max(indices[1]) + 1
    
    target_img = sample2_gray[top:bottom, left:right]
    target_img = custom_threshold(target_img, 128, 255)
    return target_img

def sample3_preprocessing(sample_path):
    img = cv2.imread(sample_path)
    img = median_filter(img, 3)
    img = img[14:-18, 10:-10]
    img_gray = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)[14:-18, 10:-10]

    h, w, c = img.shape
    red_c = img[:, :, 2]
    green_c = img[:, :, 1]
    blue_c = img[:, :, 0]

    green_dominate = (green_c > 100) & (blue_c < 180)

    mask = np.uint8(green_dominate) * 255
    indices = np.where(mask == 255)

    top = np.min(indices[0]) - 1 if np.min(indices[0]) - 1 >= 0 else 0
    bottom = np.max(indices[0]) + 1
    left = np.min(indices[1]) - 1
    right = np.max(indices[1]) + 1

    target_img = img_gray[top:bottom, left:right]
    target_img = custom_threshold(target_img, 128, 255)
    
    return target_img

def sample4_preprocessing(sample_path):
    img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    img = custom_threshold(img, 128, 255)
    img = img[20:-10, 9:]
    return img
    
def p2():
    img = cv2.imread('hw4_sample_images/TrainingSet.png', cv2.IMREAD_GRAYSCALE)
    img = custom_threshold(img, 128, 255)

    templates = generate_templates(img) # templates is a dictionary  
    
    # sample2
    sample2 = sample2_preprocessing('hw4_sample_images/sample2.png')
    characters = segment_characters(sample2, 10)
    recognized_text = recognize_characters(characters, templates)
    print('sample2:', recognized_text)

    # sample3
    sample3 = sample3_preprocessing('hw4_sample_images/sample3.png')
    characters = segment_characters_cc(sample3, 100)
    recognized_text = recognize_characters(characters, templates)
    print('sample3:', recognized_text)

    # sample4
    sample4 = sample4_preprocessing('hw4_sample_images/sample4.png')
    characters = segment_characters(sample4, 15)
    recognized_text = recognize_characters(characters, templates)
    print('sample4:', recognized_text)

if __name__ == '__main__':
    p2()