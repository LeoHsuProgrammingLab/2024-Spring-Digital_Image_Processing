import numpy as np
import cv2
from utils import *

def flood_fill(image, start_x, start_y, fill_value):
    """ Stack-based flood-fill algorithm to avoid recursion limit issues. """
    stack = [(start_x, start_y)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-connected neighbors
    while stack:
        x, y = stack.pop()
        if image[y, x] == 255:  # Only fill if the current pixel is white (part of a hole)
            image[y, x] = fill_value  # Fill the pixel with the current label
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and image[ny, nx] == 255:
                    stack.append((nx, ny))

def count_holes(img):
    """ Count holes in black objects on a white background. """

    # Start labeling holes from a unique value (e.g., 100)
    current_label = 100
    num_holes = 0

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] == 255:  # Check if the pixel is part of a potential hole
                # Perform a flood-fill to mark all pixels connected to this one
                flood_fill(img, x, y, current_label)
                current_label += 1
                num_holes += 1  # Increment the hole count for each filled region

    return num_holes

def match_template(char_img, templates):
    h, w = char_img.shape
    # show_img(char_img)
    # print(char_img)

    # get desc
    char_area, char_aspect_ratio, char_area_ratio, \
    char_vertical_hist, char_horizontal_hist, \
    char_vertical_crossings, char_horizontal_crossings = calculate_desc(char_img)
    char_holes = count_holes(char_img)
    # char_contours = find_contours(char_img)
    # char_bays = count_bays(char_contours)

    min_diff = float('inf')
    min_diff_char = ''

    for char, template in templates.items():
        # top, bottom, left, right = get_bounding_box(template)
        # template = template[top:bottom, left:right]
        # template = np.pad(template, (5, 5), 'constant', constant_values=255)
        template = cv2.resize(template, (w, h))
        # show_img(template)
        # print(template)
        # get template desc
        template_area, template_aspect_ratio, template_area_ratio, \
        template_vertical_hist, template_horizontal_hist, \
        template_vertical_crossings, template_horizontal_crossings = calculate_desc(template)
        template_holes = count_holes(template)
        
        # contours = find_contours(template)
        # template_bays = count_bays(contours)

        # print(char_holes, template_holes)
        if (char_holes != template_holes):
            # print(char, char_holes, template_holes)
            continue

        # if (char_bays != template_bays):
        #     # print(char, "bays: ", char_bays, template_bays)
        #     continue

        diff = 0
        # diff += abs(char_area - template_area)
        # diff += abs(char_aspect_ratio - template_aspect_ratio)
        # diff += abs(char_area_ratio - template_area_ratio)
        diff += np.sum(np.abs(char_vertical_hist - template_vertical_hist))
        diff += np.sum(np.abs(char_horizontal_hist - template_horizontal_hist))
        diff += np.sum(np.abs(char_vertical_crossings - template_vertical_crossings))
        diff += np.sum(np.abs(char_horizontal_crossings - template_horizontal_crossings))
        
        if diff < min_diff:
            min_diff = diff
            min_diff_char = char

    return min_diff_char

def get_bounding_box(img):
    h, w = img.shape
    # get bbx
    top, bottom, left, right = h, 0, w, 0
    for i in range(h):
        if np.any(img[i, :] == 0) and i < top:
            top = i
        if np.any(img[i, :] == 0) and i > bottom:
            bottom = i
    for i in range(w):
        if np.any(img[:, i] == 0) and i < left:
            left = i
        if np.any(img[:, i] == 0) and i > right:
            right = i
    if (bottom == 0):
        bottom = h
    if (top >= bottom):
        top = 0
    if (right == 0):
        right = w
    if (left >= right):
        left = 0

    return top, bottom, left, right

def calculate_desc(img):
    top, bottom, left, right = get_bounding_box(img)

    target_w = right - left
    target_h = bottom - top

    # cal desc
    # area
    area = np.sum(img == 0)

    # aspect ratio
    aspect_ratio = target_h / target_w

    # area ratio
    area_ratio = area / (target_h * target_w)

    # Histograms
    vertical_hist = np.sum(img == 0, axis=0) / target_w 
    horizontal_hist = np.sum(img == 0, axis=1) / target_h

    # crossing
    vertical_crossings = np.sum(np.diff(img > 0, axis=0) == 1, axis=0)
    horizontal_crossings = np.sum(np.diff(img > 0, axis=1) == 1, axis=1)

    return \
    area, aspect_ratio, area_ratio, \
    vertical_hist, horizontal_hist, \
    vertical_crossings, horizontal_crossings

def find_contours(binary_image):
    contours = []
    visited = np.zeros_like(binary_image)
    rows, cols = binary_image.shape
    
    # Directions for 8-connected neighbors
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    def dfs(x, y, contour):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cols and 0 <= ny < rows and binary_image[ny, nx] == 0 and not visited[ny, nx]:
                    visited[ny, nx] = 1
                    contour.append((nx, ny))
                    stack.append((nx, ny))
    
    for y in range(rows):
        for x in range(cols):
            if binary_image[y, x] == 0 and not visited[y, x]:
                visited[y, x] = 1
                contour = [(x, y)]
                dfs(x, y, contour)
                contours.append(contour)
    return contours

def angle_between(v1, v2):
    """ Return the angle in radians between vectors 'v1' and 'v2' """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def count_bays(contours):
    num_bays = 0
    for contour in contours:
        convexity_changes = 0
        # Convert contour list to array for processing
        points = np.array(contour)
        num_points = len(points)

        if num_points < 3:
            continue  # Not enough points to calculate direction changes

        # Calculate angle changes between consecutive points
        angles = []
        for i in range(num_points):
            p1 = points[i - 2]  # Previous point
            p2 = points[i - 1]  # Current point
            p3 = points[i]      # Next point
            
            # Create vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate the angle between vectors
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue  # Avoid division by zero
            angle = angle_between(v1, v2)
            
            # Check if the angle is a convexity change
            if angle > np.pi / 2:
                angles.append(angle)

        # Filter and count significant changes
        for i in range(1, len(angles)):
            if angles[i] != angles[i - 1]:
                convexity_changes += 1
        
        # If more than one convexity change, count it as a bay
        if convexity_changes > 1:
            num_bays += 1

    return num_bays

