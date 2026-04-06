import cv2

image = cv2.imread("sample.jpg", 0)

if image is None:
    print("Image not found. Make sure 'sample.jpg' exists in the same folder.")
else:
    _, threshold_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresholded Image", threshold_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
