import cv2

image = cv2.imread("sample.jpg")

if image is None:
    print("Image not found. Make sure 'sample.jpg' exists in the same folder.")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    cv2.imshow("Edge Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
