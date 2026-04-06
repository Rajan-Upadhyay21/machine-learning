import cv2
import matplotlib.pyplot as plt

image = cv2.imread("sample.jpg", 0)

if image is None:
    print("Image not found. Make sure 'sample.jpg' exists in the same folder.")
else:
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.title("Image Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()
