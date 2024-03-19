import cv2
import matplotlib.pyplot as plt
import numpy as np

def bucket_filling(img):
    x, y = img.shape[0], img.shape[1]
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    print(np.sum(hist))

    # for i in range(x):
    #     for i in range(y):
    #         pass

def plot_histogram(hist):
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    img = cv2.imread('/Users/leohsuinthehouse/Desktop/ID.jpg')
    bucket_filling(img)