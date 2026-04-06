import cv2

image = cv2.imread("sample.jpg")

if image is None:
    print("Image not found. Make sure 'sample.jpg' exists in the same folder.")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    cv2.imshow("Contours", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
