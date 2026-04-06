import cv2

image = cv2.imread("sample.jpg")

if image is None:
    print("Image not found. Make sure 'sample.jpg' exists in the same folder.")
else:
    resized_image = cv2.resize(image, (300, 300))
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
