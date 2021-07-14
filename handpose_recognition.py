import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands( min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
	while cap.isOpened():
		flag, image = cap.read()
		if not flag:
			print('something is wrong with camera')
			break


		image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
		

		image.flags.writeable = False
		results = hands.process(image)
		
		results = hands.process(image)

		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				mp_drawing.draw_landmarks( image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


		cv2.imshow('live', image)

		if cv2.waitKey(10) & 0xff == ord('q'):
			break



cap.release()
cv2.destroyAllWindows()