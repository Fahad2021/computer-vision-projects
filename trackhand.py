import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()




# cvzone version 1.5.1
# from cvzone.HandTrackingModule import HandDetector
# import cv2

# cap = cv2.VideoCapture(0)
# detector = HandDetector(detectionCon=0.8, maxHands=2)
# while True:
#     # Get image frame
#     success, img = cap.read()
#     # Find the hand and its landmarks
#     hands, img = detector.findHands(img)  # with draw
#     # hands = detector.findHands(img, draw=False)  # without draw

#     if hands:
#         # Hand 1
#         hand1 = hands[0]
#         lmList1 = hand1["lmList"]  # List of 21 Landmark points
#         bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
#         centerPoint1 = hand1['center']  # center of the hand cx,cy
#         handType1 = hand1["type"]  # Handtype Left or Right

#         fingers1 = detector.fingersUp(hand1)

#         if len(hands) == 2:
#             # Hand 2
#             hand2 = hands[1]
#             lmList2 = hand2["lmList"]  # List of 21 Landmark points
#             bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
#             centerPoint2 = hand2['center']  # center of the hand cx,cy
#             handType2 = hand2["type"]  # Hand Type "Left" or "Right"

#             fingers2 = detector.fingersUp(hand2)

#             # Find Distance between two Landmarks. Could be same hand or different hands
#             length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # with draw
#             # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
#     # Display
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)
# cap.release()
# cv2.destroyAllWindows()