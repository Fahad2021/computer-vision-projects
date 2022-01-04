import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # Convert frame to image
    image=frame
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(image_gray, (5,5), 0)
    #cv2.imshow('Gaussian Blur',gaussian_blur)
    
    canny_edges = cv2.Canny(gaussian_blur, 50, 150)
    cv2.imshow('Canny Edge Detection',canny_edges)
    
    ret, thresh_image = cv2.threshold(canny_edges, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Sketch Image',thresh_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()   