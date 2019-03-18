from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
cam = cv2.VideoCapture(1)

past_values_x = []
def min_intensity_x(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	min_sum_y = 255 * len(img)
	min_index_x = -1

	for x in range(len(img[0])):

		temp_sum_y = 0

		for y in range(len(img)):
			temp_sum_y += img[y][x]

		if temp_sum_y < min_sum_y:
			min_sum_y = temp_sum_y
			min_index_x = x

	past_values_x.append(min_index_x)

	if len(past_values_x) > 3:
		past_values_x.pop(0)

	return int(sum(past_values_x) / len(past_values_x))

past_values_y = []
def min_intensity_y(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	min_sum_x = 255 * len(img[0])
	min_index_y = -1

	for y in range(len(img)):

		temp_sum_x = 0

		for x in range(len(img[0])):
			temp_sum_x += img[y][x]

		if temp_sum_x < min_sum_x:
			min_sum_x = temp_sum_x
			min_index_y = y

	past_values_y.append(min_index_y)

	if len(past_values_y) > 3:
		past_values_y.pop(0)

	return int(sum(past_values_y) / len(past_values_y))

def extract_eye(image, left, bottom_left, bottom_right, right, upper_right, upper_left):
	lower_bound = max([left[1], right[1], bottom_left[1], bottom_right[1], upper_left[1], upper_right[1]])
	upper_bound = min([left[1], right[1], upper_left[1], upper_right[1], bottom_left[1], bottom_right[1]])

	eye = image[upper_bound-3:lower_bound+3, left[0]-3:right[0]+3]

	pupil_x = min_intensity_x(eye)
	pupil_y = min_intensity_y(eye)

	cv2.line(eye,(pupil_x,0),(pupil_x,len(eye)),(0,255,0), 1)
	cv2.line(eye,(0,pupil_y),(len(eye[0]),pupil_y),(0,255,0), 1)

	cv2.line(image,(int((bottom_left[0] + bottom_right[0]) / 2), lower_bound), (int((upper_left[0] + upper_right[0]) / 2), upper_bound),(0,0,255), 1)
	cv2.line(image,(left[0], left[1]), (right[0], right[1]),(0,0,255), 1)

	image[upper_bound-3:lower_bound+3, left[0]-3:right[0]+3] = eye
	#centerx = int(( + right[0])/2)
	#centery = int((left[1] + right[1])/2)
	return eye, pupil_x, pupil_y

def mapper(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return int(rightMin + (valueScaled * rightSpan))

#def crosshair(img, src, dst, color:


while(True):
	# load the input image, resize it, and convert it to grayscale
	_, image = cap.read()
	_, frame = cam.read()
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	control_shape = frame.shape
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		count = 1
		eye_out, centerx, centery = extract_eye(image, shape[36], shape[41], shape[40], shape[39], shape[38], shape[37])
		right_eye = imutils.resize(eye_out, width=100, height=50)
		cv2.circle(image, (centerx, centery), 10, (255,255,255), 1)
		for (x, y) in shape:
			if count > 36 and count < 43:
					cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

			count += 1

		image[0:len(right_eye),0:len(right_eye[0])] = right_eye
		eye_shape = right_eye.shape

		#print('CenterY : ', centery)
		#print(eye_shape[0], ' : ', eye_shape[1])
		controlx = int(control_shape[1]/2) + int(2.8*mapper(centerx,0, eye_shape[1]-45, -1*int(control_shape[1]/2), int(control_shape[1]/2)))
		controly = int(control_shape[0]/2) + int(2*mapper(centery, 0, 8, -1*int(control_shape[0]/2)-50, int(control_shape[1]/2)))+20
		#print('ControlY : ', controly)
		tracked = [control_shape[0]-controlx, controly]
		#Limits
		if (tracked[0] < 0): tracked[0] = 0
		elif (tracked[0] > control_shape[1]): tracked[0] = control_shape[1]
		if (tracked[1] < 0): tracked[1] = 1
		elif (tracked[1] > control_shape[0]): tracked[1] = control_shape[0]
		#Convert to tuple
		tracked = (tracked[0], tracked[1])
		#Display location
		cv2.circle(frame,tracked, 10, (255,255,255), -1)
		#Find the quadrant on key press
		location = None
		if cv2.waitKey(1) & 0xFF == ord('s'):
			#Center
			if (tracked[0] > int(control_shape[1]/2)-20) and (tracked[0] < int(control_shape[1]/2)+20):
				location = 'center'
				os.system("espeak 'Center'")
			#Top_Left
			elif (tracked[0] < int(control_shape[1]/2)-20) and (tracked[1] < int(control_shape[0]/2)):
				location = 'top left'
				os.system("espeak 'Top Left'")
			#Top_Right
			elif (tracked[0] > int(control_shape[1]/2)+20) and (tracked[1] < int(control_shape[0]/2)):
				location = 'top right'
				os.system("espeak 'Top Right'")
			#Bottom_Left
			elif (tracked[0] < int(control_shape[1]/2)-20) and (tracked[1] > int(control_shape[0]/2)):
				location = 'bottom left'
				os.system("espeak 'Bottom Left'")
			#Bottom_Right
			elif (tracked[0] > int(control_shape[1]/2)+20) and (tracked[1] > int(control_shape[0]/2)):
				location = 'bottom right'
				os.system("espeak 'Bottom Right'")

	cv2.imshow('Control', frame)
	cv2.imshow("Pupil Tracking", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# show the output image with the face detections + facial landmarks

cv2.waitKey(0)
