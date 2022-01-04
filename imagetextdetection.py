import cv2
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import pyttsx3
speak=pyttsx3.init()
img=cv2.imread('2.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
dem=pytesseract.image_to_string(img)
print(dem)
# print(pytesseract.image_to_boxes(img))

# detecting character and bound it boxes
# hImg,wImg,_ = img.shape
# boxes=pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#     print(b)
#     b=b.split(' ')
#     print(b)
#     x,y,w,h=int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),3)
#     cv2.putText(img,b[0],(x,hImg-y+25),cv2.FORMATTER_FMT_MATLAB,1,(50,50,255),2)
# detecting words
hImg,wImg,_ = img.shape
boxes=pytesseract.image_to_data(img)
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b=b.split()
        # print(b)
        if len(b)==12:
            x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),3)
            cv2.putText(img,b[11],(x,y),cv2.FORMATTER_FMT_CSV,1,(50,50,255),2)
            speak.say(dem)
            speak.runAndWait()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imshow('Result',img)
cv2.waitKey(0)
