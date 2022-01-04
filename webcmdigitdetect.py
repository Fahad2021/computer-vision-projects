import cv2
import pytesseract
import numpy as np
import pyttsx3
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
speak=pyttsx3.init()

# webcam
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
def captureScreen(bbox=(300,300,1500,1000)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    return capScr
while True:
    timer = cv2.getTickCount()
    _,img = cap.read()
    #img = captureScreen()
    #DETECTING CHARACTERES
    #DETECTING WORD
    hImg, wImg,_ = img.shape
    boxes=pytesseract.image_to_data(img)
    
    for x,b in enumerate(boxes.splitlines()):
        if x!=0:
            b=b.split()
            # print(b)
            if len(b)==12:
                x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),3)
                cv2.putText(img,b[11],(x,y),cv2.FORMATTER_FMT_CSV,1,(50,50,255),2)
                dem=pytesseract.image_to_string(img)
                print(dem)
                speak.say(dem)
                speak.runAndWait()
                
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    cv2.putText(img, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,230,20), 2);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("Result",img)
    cv2.waitKey(1)

cv2.imshow('Result',img)
cv2.waitKey(0)


