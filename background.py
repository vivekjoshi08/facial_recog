import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

print('Enter the background you want \nPress 1 for arc_bridge \nPress 2 for aurora \nPress 3 for home interior \nPress 4 for nature \nPress 5 for office \nPress 6 for deadpool')

inp = int(input())

if inp == 1 :
	bg_image = cv2.imread('images/arc_bridge.jpeg')

elif inp == 2:
	bg_image = cv2.imread('images/aurora.jpg')

elif inp == 3:
	bg_image = cv2.imread('images/home_int.jpg')

elif inp == 4:
	bg_image = cv2.imread('images/nature1.jpg')

elif inp == 5:
	bg_image = cv2.imread('images/office.jpg')

elif inp == 6:
	bg_image = cv2.imread('images/deadpool.jpg')
else :
	bg_image = cv2.imread('images/solid_white.jpg')

model = mp_selfie_segmentation.SelfieSegmentation(model_selection = 1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
	flag, frame = cap.read()
	if not flag:
		print('Error')
		break
	
	results = model.process(frame)

	condition = np.stack((results.segmentation_mask,)*3, axis =- 1) > 0.1
	if bg_image is None:
		print('something went wrong try again')
		break

	bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
	output_image = np.where(condition, frame, bg_image)

	cv2.imshow('Frame', output_image)
	if cv2.waitKey(10) & 0xff == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
