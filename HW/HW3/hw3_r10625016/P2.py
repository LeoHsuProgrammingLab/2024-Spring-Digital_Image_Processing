import cv2
import numpy as np
from P1 import *

# P2_a
def convolution(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=(0, 0))
    
    conv_img = np.zeros_like(img).astype(np.float32)

    for i in range(h):
        for j in range(w):
            patch = padded_img[i:i+kh, j:j+kw]
            conv_img[i, j] = np.sum(patch * kernel)
    return conv_img

def laws_method(img):
    L3 = np.array([1, 2, 1])
    E3 = np.array([-1, 0, 1])
    S3 = np.array([-1, 2, -1])
    basic_kernels = []
    for i in [L3, E3, S3]:
        for j in [L3, E3, S3]:
            basic_kernels.append(np.outer(i, j))

    normalized_imgs, feature_imgs = [], []
    for kernel in basic_kernels:
        feat = convolution(img, kernel)
        normalized_feat = (feat - np.min(feat)) / (np.max(feat) - np.min(feat)) * 255
        normalized_imgs.append(normalized_feat)
        feature_imgs.append(feat)
    
    return feature_imgs, normalized_imgs

def compute_energy(imgs, kernel_size=13):
    pad_h, pad_w = kernel_size // 2, kernel_size // 2
    energy_imgs = []

    for img in imgs:
        h, w = img.shape
        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=(0, 0))
        energy_img = np.zeros_like(img).astype(np.float32)
        for i in range(h):
            for j in range(w):
                patch = padded_img[i:i+kernel_size, j:j+kernel_size]
                energy_img[i, j] = np.sum(patch ** 2)
        normalized_energy_img = (energy_img - np.min(energy_img)) / (np.max(energy_img) - np.min(energy_img)) * 255
        # energy_img
        energy_imgs.append(normalized_energy_img)
    
    return energy_imgs

def p2_a():
    img = cv2.imread('hw3_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
    # feature extraction
    feature_imgs, normalized_imgs = laws_method(img)
    for i, feat in enumerate(normalized_imgs):
        cv2.imwrite(f'feature{i+1}.png', feat)

    energy_imgs = compute_energy(normalized_imgs)
    for i, energy in enumerate(energy_imgs):
        cv2.imwrite(f'energy{i+1}.png', energy)

# P2_b
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))   

def kmeans(features, k=3, max_iters=10):
    # Randomly initialize the centroids as k random samples from features
    centroids = features[np.random.choice(features.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # Assign samples to nearest centroids
        clusters = np.array([np.argmin([euclidean_distance(feature, centroid) for centroid in centroids]) for feature in features])

        # Calculate new centroids from the clusters
        new_centroids = np.array([features[clusters == i].mean(axis=0) for i in range(k)])

        # If centroids don't change, break out of the loop
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids

def p2_b():
    img = cv2.imread('hw3_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
    
    feature_imgs, normalized_imgs = laws_method(img)
    energy_imgs = compute_energy(normalized_imgs)
    kmeans_input = np.stack(energy_imgs, axis=-1).reshape(-1, len(energy_imgs))
    k = 4
    clusters, centroids = kmeans(kmeans_input, k=k, max_iters=5)
    labeled_img = clusters.reshape(img.shape)
    # connected component labeling
    # thresholding for each cluster: if lower than set to 0
    colors = [
        [255, 0, 0], # red
        [0, 255, 0], # green
        [0, 0, 255], # blue
        [255, 255, 0], # yellow
        [0, 255, 255] # cyan
    ]
    result_img = np.zeros((img.shape[0], img.shape[1], 3))

    for i in range(5):
        result_img[labeled_img == i] = colors[i]
    
    cv2.imwrite('result4.png', result_img)

# P2_c
def gen_checkboard_texture(h, w, tile_size=32):
    checkboard = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if (i // tile_size) % 2 == 0:
                if (j // tile_size) % 2 == 0:
                    checkboard[i, j] = 255
            else:
                if (j // tile_size) % 2 == 1:
                    checkboard[i, j] = 255
    return checkboard

def gen_sinusoidal_texture(h, w, freq=0.1):
    sinusoidal = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            sinusoidal[i, j] = 127.5 * np.sin(2 * np.pi * freq * i) + 127.5
    return sinusoidal

def gen_noise_texture(h, w):
    noise = np.random.randint(0, 256, (h, w))
    return noise

def gen_rain_texture(height, width, line_length=20, line_width=2, line_color=255, background=0, density=0.005):
    texture = np.full((height, width), background, dtype=np.uint8)
    num_lines = int(density * width * height)

    for _ in range(num_lines):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        end_y = y + line_length if (y + line_length < height) else height - 1
        cv2.line(texture, (x, y), (x, end_y), line_color, line_width)

    return texture

def p2_c():
    img = cv2.imread('result4.png')
    colors = [
        [255, 0, 0], # red
        [0, 255, 0], # green
        [0, 0, 255], # blue
        [255, 255, 0], # yellow
        [0, 255, 255] # cyan
    ]
    masks = [np.all(img == color, axis=-1) for color in colors]

    checkboard = gen_checkboard_texture(img.shape[0], img.shape[1])
    sinusoidal = gen_sinusoidal_texture(img.shape[0], img.shape[1])
    noise = gen_noise_texture(img.shape[0], img.shape[1])
    rain = gen_rain_texture(img.shape[0], img.shape[1])

    textures = [checkboard, sinusoidal, noise, rain]
    result_img = np.zeros((img.shape[0], img.shape[1], 3))
    k = 3
    for i in range(k):
        texture = textures[i]
        mask = masks[i]

        for c in range(3):
            result_img[mask, c] = texture[mask]
    
    cv2.imwrite('result5.png', result_img)

# P2_d
def calculate_surface_err_mtx(block1, block2):
    return (block1 - block2) ** 2

def find_minimum_cost_seam(block1, block2, vertical: bool):
    surface_err = calculate_surface_err_mtx(block1, block2)

    if not vertical:
        surface_err = surface_err.T

    h, w = surface_err.shape
    dp_table = np.zeros((h, w))
    dp_table[0] = surface_err[0]

    for i in range(1, h):
        for j in range(w):
            min_prev = dp_table[i-1, j]
            if j > 0:
                min_prev = min(min_prev, dp_table[i-1, j-1])
            if j < w - 1:
                min_prev = min(min_prev, dp_table[i-1, j+1])
            dp_table[i, j] = surface_err[i, j] + min_prev
    
    # Backtrack to find the seam path
    seam = np.zeros(h, dtype=int)
    seam[-1] = np.argmin(dp_table[-1]) # get the id of the minimum cost pixel in the last row
    
    for i in range(h-2, -1, -1):
        j = seam[i+1] # i+1 is the next row
        min_prev = dp_table[i, j]
        seam[i] = j
        if j > 0 and dp_table[i, j-1] < min_prev:
            min_prev = dp_table[i, j-1]
            seam[i] = j - 1
        if j < w - 1 and dp_table[i, j+1] < min_prev:
            seam[i] = j + 1

    return seam

def blend_blocks(block1, block2, seam, overlap, vertical: bool):
    h, w = block1.shape
    print(seam)
    result = np.zeros_like(block1) # result block: already expanded
    if vertical: # blend horizontally to left: seam is vertical
        for i in range(h):
            j = seam[i]
            result[i, :j] = block1[i, :j]
            result[i, j:] = block2[i, j:]
    else: # blend horizontally to top: seam is horizontal
        for j in range(w):
            i = seam[j]
            result[:i, j] = block1[:i, j]
            result[i:, j] = block2[i:, j]

    return result

def quilt_image(source_img, output_size, block_size):
    overlap = block_size // 6
    new_height, new_width = output_size
    output_img = np.zeros((new_height + block_size, new_width + block_size))

    y = 0
    while y < new_height:
        x = 0
        while x < new_width:
            origin_y = np.random.randint(0, source_img.shape[0] - block_size)
            origin_x = np.random.randint(0, source_img.shape[1] - block_size)
            block = source_img[origin_y:origin_y+block_size, origin_x:origin_x+block_size]

            if x == 0:
                output_img[y:y+block_size, x:x+block_size] = block
                x += block_size
                continue

            if x > 0:
                left_block = output_img[y:y+block_size, x-overlap:x]
                block_overlap = block[:, :overlap]
                vertical_seam = find_minimum_cost_seam(left_block, block_overlap, vertical=True)
                block = blend_blocks(output_img[y:y+block_size, x-overlap:x+block_size-overlap], block, vertical_seam, overlap, vertical=True)

                output_img[y:y+block_size, x-overlap:x+block_size-overlap] = block

            if y > 0:
                top_block = output_img[y-overlap:y, x:x+block_size]
                block_overlap = block[:overlap, :]
                horizontal_seam = find_minimum_cost_seam(top_block, block_overlap, vertical=False)
                block = blend_blocks(output_img[y-overlap:y+block_size-overlap, x:x+block_size], block, horizontal_seam, overlap, vertical=False)

                output_img[y-overlap:y+block_size-overlap, x:x+block_size] = block

            x += block_size - overlap
        y += block_size - overlap
    
    return output_img[:new_height, :new_width]

def p2_d():
    img = cv2.imread('hw3_sample_images/sample3.png', cv2.IMREAD_GRAYSCALE)
    block_size = 512
    new_height, new_width = img.shape[0] * 2, img.shape[1] * 2
    quilted_img = quilt_image(img, (new_height, new_width), block_size)
    cv2.imwrite('result6.png', quilted_img)

if __name__ == "__main__":
    # p2_a()
    # p2_b()
    # p2_c()
    p2_d()