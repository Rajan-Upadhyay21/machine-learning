import cv2

image = cv2.imread("sample.jpg")

if image is None:
    print("Image not found. Make sure 'sample.jpg' exists in the same folder.")
else:
    flipped_horizontal = cv2.flip(image, 1)
    flipped_vertical = cv2.flip(image, 0)
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow("Original", image)
    cv2.imshow("Flipped Horizontal", flipped_horizontal)
    cv2.imshow("Flipped Vertical", flipped_vertical)
    cv2.imshow("Rotated 90 Degrees", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
