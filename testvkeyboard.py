import cv2
from cvzone.HandTrackingModule import HandDetector
cap=cv2.VideoCapture(0)

detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    # img=detector.findHands(img)
    hands, img = detector.findHands(img) 
    from cvzone.HandTrackingModule import HandDetector


    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right
        fingers1 = detector.fingersUp(hand1)
        # if len(hands) == 2:q
        #     # Hand 2
        #     hand2 = hands[1]
        #     lmList2 = hand2["lmList"]  # List of 21 Landmark points
        #     bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
        #     centerPoint2 = hand2['center']  # center of the hand cx,cy
        #     handType2 = hand2["type"]  # Hand Type "Left" or "Right"
        #     fingers2 = detector.fingersUp(hand2)
        #     # Find Distance between two Landmarks. Could be same hand or different hands
        #     length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
        #     # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
        if cv2.waitKey(5) & 0xFF == 27:
          break