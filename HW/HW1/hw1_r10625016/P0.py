import cv2

# (a)
def p0_a(filename):
    img = cv2.imread(filename)
    height, width = img.shape[:2]
    # for i in range(width):
    #     for j in range(int(height/2)):
    #         temp = img[j, i].copy()
    #         img[j, i] = img[height - j - 1, i]
    #         img[height - j - 1, i] = temp
    img = img[::-1, :, :]

    cv2.imwrite('result1.png', img)

    return img
# (b)
def p0_b(filename):
    img = cv2.imread(filename) # B, G, R
    # img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for row in range(img.shape[0]): 
        for col in range(img.shape[1]):
            # img[row, col] = sum(img[row, col]) / 3
            img[row, col] = img[row, col][2] * 0.299 + img[row, col][1] * 0.587 + img[row, col][0] * 0.114
    cv2.imwrite('result2.png', img)

if __name__ == "__main__":
    p0_a('hw1_sample_images/sample1.png')
    p0_b('result1.png')