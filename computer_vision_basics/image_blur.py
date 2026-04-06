import cv2

image = cv2.imread("sample.jpg")

if image is None:
    print("Image not found. Make sure 'sample.jpg' exists in the same folder.")
else:
    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imshow("Blurred Image", blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
