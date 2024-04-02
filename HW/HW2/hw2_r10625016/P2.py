import cv2
import numpy as np
import sys
MAX_INT = sys.maxsize

def image2cartesian(img, p, q):
    P, Q = img.shape # P: height, Q: width 
    u = q
    v = P - p
    return u, v

# def cartesian2image(img, x, y): the same as image2cartesian

# (j, k) -> (u, v) 
# k, j: cartesian coordinate
def backward_translation(img, k, j, tx, ty): 
    P, Q = img.shape
    u = k - tx
    v = j - ty - (P-P) # (P - J)
    p, q = image2cartesian(img, u, v)
    return p, q

def forward_translation(img, p, q, tx, ty):
    P, Q = img.shape
    u, v = image2cartesian(img, p, q)
    k = u + tx
    j = v + ty + (P-P)
    return k, j

def bilinear_interpolation(img, p, q):
    P, Q = img.shape
    p1, q1 = int(p), int(q)
    p2, q2 = p1 + 1, q1 + 1
    if p2 >= P:
        p2 = P - 1
    if q2 >= Q:
        q2 = Q - 1
    a = p - p1
    b = q - q1
    value = (1 - a) * (1 - b) * img[p1, q1] + a * (1 - b) * img[p2, q1] + (1 - a) * b * img[p1, q2] + a * b * img[p2, q2]
    return value

def scaling(img, p, q, sx, sy):
    P, Q = img.shape
    u, v = image2cartesian(img, p, q)
    x = u / sx
    y = v / sy
    return x, y

def rotate(img, p, q, angle):
    P, Q = img.shape
    u, v = image2cartesian(img, p, q)
    x = u * np.cos(angle) - v * np.sin(angle)
    y = u * np.sin(angle) + v * np.cos(angle)
    return x, y

def rotate_matrix(angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    R = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    return R

def translation_matrix(tx, ty):
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    return T

def scaling_matrix(sx, sy):
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])
    return S

def linear_transformation(img, p, q, angle, tx = 0, ty = 0, sx = 1, sy = 1):
    # fisrt
    T = translation_matrix(tx, ty)
    # second: if the rotation is around the origin, otherwise, we need to translate the image to the p, q first
    T_o = translation_matrix(-p, -q)
    R = rotate_matrix(angle)
    T_i = translation_matrix(p, q)
    # third
    S = scaling_matrix(sx, sy)

    # combine the transformation
    H = np.dot(S, np.dot(T_i, np.dot(R, np.dot(T_o, T))))
    
    return H
    
def barrel_distortion(img, pivot_x, pivot_y, k):
    rows, cols = img.shape

    # New image array with the same shape but with a 4 channel (RGBA)
    distorted_image = np.zeros_like(img)

    # Find the center of the image
    cx, cy = pivot_x, pivot_y

    # Iterate through every pixel in the image
    for x in range(rows):
        for y in range(cols):
            # Normalize x and y coordinates to be around the center
            dx = (x - cx) / cx
            dy = (y - cy) / cy
            # Calculate the square distance from the center
            d = dx * dx + dy * dy
            # Calculate the factor of distortion based on the distance
            factor = 1 + k * d

            # Get the new pixel location after distortion
            new_x, new_y = cx + cx * factor * dx, cy + cy * factor * dy
            new_x, new_y = int(new_x), int(new_y)

            # Copy pixel from original image if it falls within the image size
            if 0 <= new_x < rows and 0 <= new_y < cols:
                distorted_image[x, y] = img[new_x, new_y]
            else:
                distorted_image[x, y] = 255

    return distorted_image

# ref: https://zh.wikipedia.org/zh-tw/%E9%B1%BC%E7%9C%BC%E9%95%9C%E5%A4%B4
def fisheye_correction(img, strength, center_scale_factor, pivot_x, pivot_y, distortion_type='stereographic'):
    height, width = img.shape
    distorted_img = np.zeros_like(img)

    # Define parameters
    cx, cy = pivot_x, pivot_y  # Center of distortion

    # Generate mapping function
    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            r = np.sqrt(dx**2 + dy**2)  # Distance from the center

            if distortion_type == 'equidistant':
                # Equidistant projection model
                theta = np.arctan(r / strength)
                scale = strength * theta / r if r != 0 else 1.0
            elif distortion_type == 'stereographic':
                # Stereographic projection model
                scale = 2 * np.arctan(r / (2 * strength)) / np.pi
            else:
                raise ValueError("Invalid distortion_type. Choose 'equidistant' or 'stereographic'.")
            
            scale *= center_scale_factor * (1 - np.exp(-r / (2 * strength)))

            # Apply mapping
            new_x = int(cx + scale * dx)
            new_y = int(cy + scale * dy)

            # Interpolation
            if 0 <= new_x < width and 0 <= new_y < height:
                distorted_img[y, x] = img[new_y, new_x]

    return distorted_img

def get_target_pixel(img): # check if the pixel value is not 0
    h, w = img.shape
    x_min, x_max, y_min, y_max = MAX_INT, 0, MAX_INT, 0
    limit_point = [[], [], [], []]

    for i in range(h):
        for j in range(w):
            if img[i, j] != 254:
                if i < y_min:
                    y_min = i
                    limit_point[2] = [i, j]
                if i > y_max:
                    y_max = i
                    limit_point[3] = [i, j]
                if j < x_min:
                    x_min = j
                    limit_point[0] = [i, j]
                if j > x_max:
                    x_max = j
                    limit_point[1] = [i, j]

    limit = [x_min, x_max, y_min, y_max]   

    return limit, limit_point

  
def p2_a():
    img = cv2.imread("hw2_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE)
    new_img = np.full_like(img, 254)

    h, w = img.shape
    limit, limit_point = get_target_pixel(img) # [x_min, x_max, y_min, y_max]
    
    pivot_i, pivot_j = limit_point[3]
    H = linear_transformation(img, pivot_i, pivot_j, -45, 0, 0, 1.1, 1.1)
    T = translation_matrix(-60, -40)
    H_inv = np.linalg.inv(np.dot(T, H))


    for i in range(h):
        for j in range(w):
            coords = np.dot(H_inv, np.array([i, j, 1]))
            x, y = coords[0], coords[1]
            if x < 0 or x >= h or y < 0 or y >= w:
                continue
            value = bilinear_interpolation(img, x, y)
            new_img[i, j] = value

    # fish eye
    limit, limit_point = get_target_pixel(new_img)
    x_min, x_max, y_min, y_max = limit
    pivot_x = (x_min + x_max) // 2
    pivot_y = (y_min + y_max) // 2

    new_img = fisheye_correction(new_img, 50, 1.6, pivot_x + 100, pivot_y)
    new_img[new_img == 0] = 254

    # R = rotate_matrix(40)
    T = translation_matrix(-40, -20)
    # H = np.dot(R, T)
    H_inv = np.linalg.inv(T)

    for i in range(h):
        for j in range(w):
            coords = np.dot(H_inv, np.array([i, j, 1]))
            x, y = coords[0], coords[1]
            if x < 0 or x >= h or y < 0 or y >= w:
                continue
            value = bilinear_interpolation(new_img, x, y)
            new_img[i, j] = value

    cv2.imwrite("result8.png", new_img)

def get_segment_line(img):
    h, w = img.shape
    threshold = 250
    borders = []

    for i in range(w):
        if not np.all(img[:, i] >= threshold):
            print(i, img[:, i])
            if (np.all(img[:, i-1] >= threshold) or np.all(img[:, i+1] >= threshold)):
                borders.append(i)

    # print(borders, len(borders))
    # [56, 166, 199, 309, 352, 462, 494, 605, 616, 631, 642, 753]
    return borders

def apply_wave_transformation(img, canvas_img, x_base, amplitude, frequency):
    h, w = img.shape
    # Create a new image with the same size and type as the original one, filled with zeros (black)
    limit_h, limit_w = canvas_img.shape

    for i in range(h):
        for j in range(w):
            # Calculate the new y coordinate based on the wave function
            j_new = int(j + amplitude * np.sin(2 * np.pi * frequency * i / h))

            # Map the original pixel to the new position
            j_new = j_new + x_base
            if i < limit_h and j_new < limit_w:
                canvas_img[i, j_new] = img[i, j]

def p2_b():
    img = cv2.imread("hw2_sample_images/sample5.png", cv2.IMREAD_GRAYSCALE)
    # borders = get_segment_line(img)
    borders = [166, 199, 309, 352, 462, 494, 605, 642]
    segment_line = [ (borders[i] + borders[i+1]) // 2 for i in range(0, len(borders), 2)]
    
    new_img = np.full_like(img, 254)
    h, w = img.shape

    segment_line = [0] + segment_line + [w]

    for i in range(len(segment_line) - 1):
        left = segment_line[i]
        right = segment_line[i+1]
        cropped_img = img[:, left:right]
        limit, limit_point = get_target_pixel(cropped_img)
        x_min, x_max, y_min, y_max = limit

        apply_wave_transformation(cropped_img, new_img, left, 21, 4.7)

    cv2.imwrite("result9.png", new_img)

if __name__ == "__main__":
    p2_a()
    p2_b()