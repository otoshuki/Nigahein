#Import libraries
import cv2.aruco as aruco
import numpy as np
import imutils
import dlib
import cv2
import os
import time
import serial
import struct

#Intitlize the face detector
detector = dlib.get_frontal_face_detector()
#Intitialize the face landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#Turn on the webuser_cams
face_cam = cv2.Videoface_camture(1)
user_cam = cv2.Videoface_camture(2)
#Set up the serial port-port can change
arduino = serial.Serial('/dev/ttyACM0', 9600)
#Set up the dictionary for Aruco marker
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
#Get the params for the dictionary
params = aruco.DetectorParameters_create()
#Global variables for control signal
delay = 0
trigger = 0
#Global variables to store pupil location history
past_values_x = []
past_values_y = []

def min_intensity_x(img):
	"""
	Credits to Tobias Roeddiger - https://github.com/TobiasRoeddiger/
	A function to find the pupil location along x axis by finding minimum intensities
	Input: img: the cropped eye
	Output: returns the pupil location along x
	"""
	#Convert to grayscale
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#Set maximum as initial guess
	min_sum = 255 * len(img)
	min_index = -1
	for x in range(len(img[0])):
		temp_sum = 0
		#Sum up all the intensities along x
		for y in range(len(img)):
			temp_sum += img[y][x]
		#Select the minimum intensity as the present guess
		if temp_sum < min_sum:
			min_sum = temp_sum
			min_index = x
	#Append to the previously determined values
	past_values_x.append(min_index)
	#Remove values more than 3 frames old
	if len(past_values_x) > 3:
		past_values_x.pop(0)
	#Get the average along the readings to get better accuracy
	return int(sum(past_values_x) / len(past_values_x))

def min_intensity_y(img):
	"""
	Credits to Tobias Roeddiger - https://github.com/TobiasRoeddiger/
	A function to find the pupil location along y axis by finding minimum intensities
	Input: img: the cropped eye
	Output: returns the pupil location along y
	"""
	#Convert to grayscale
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#Set maximum as initial guess
	min_sum = 255 * len(img[0])
	min_index = -1
	for y in range(len(img)):
		temp_sum = 0
		#Sum up all the intensities along x
		for x in range(len(img[0])):
			temp_sum += img[y][x]
		#Select the minimum intensity as the present guess
		if temp_sum < min_sum:
			min_sum = temp_sum
			min_index = y
	#Append to the previously determined values
	past_values_y.append(min_index)
	#Remove values more than 3 frames old
	if len(past_values_y) > 3:
		past_values_y.pop(0)
	#Get the average along the readings to get better accuracy
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

def conv2polar(coor):
	r = int(np.sqrt(coor[0]*coor[0] + coor[1]*coor[1]))
	theta = int(np.arctan2(coor[1], coor[0])*180/3.14)
	polar = (r, theta)
	return polar

def sector(theta):
	sec = 0
	if (theta > -22.5) and (theta < 22.5): sec = 1 #sector1
	elif (theta > 22.5) and (theta < 67.5): sec = 2 #sector2
	elif (theta > 67.5) and (theta < 112.5): sec = 3 #sector3
	elif (theta > 112.5) and (theta < 157.5): sec = 4 #sector4
	elif (theta > 157.5) and (theta < 180): sec = 5 #sector5
	elif (theta > -180) and (theta < -157.5): sec = 5 #sector5
	elif (theta > -157.5) and (theta < -112.5): sec = 6 #sector6
	elif (theta > -112.5) and (theta < -67.5): sec = 7 #sector7
	elif (theta > -67.5) and (theta < -22.5): sec = 8 #sector8
	return sec

while(True):
	# load the input image, resize it, and convert it to grayscale
	_, image = face_cam.read()
	image = np.rot90(image, 2)
	_, frame = user_cam.read()
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	corners, ids, rejected = aruco.detectMarkers(gray2, aruco_dict,parameters = params)
	# detect faces in the grayscale image
	detected_faces = detector(gray, 1)
	control_shape = frame.shape
	#Detect ArUco marker
	detected = aruco.drawDetectedMarkers(frame, corners)
	marker_sector = 0
	eye_sector = 0
	if np.all(ids != None):
		#print(corners[0][0])
		marker_x = int((corners[0][0][0][0] + corners[0][0][2][0])/2)
		marker_y = int((corners[0][0][0][1] + corners[0][0][2][1])/2)
		cv2.circle(detected, (marker_x, marker_y), 5, (255,0,0), -1)
		marker_coor = (marker_x - int(control_shape[1]/2), marker_y - int(control_shape[0]/2))
		marker_polar = conv2polar(marker_coor)
		marker_sector = sector(marker_polar[1])
	# loop over the face detections
	for (i, rect) in enumerate(detected_faces):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = imutils.face_utils.shape_to_np(shape)
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
		#Origin shift
		shifted = (tracked[0]-(control_shape[1]/2), tracked[1]-(control_shape[0]/2))
		#Convert to polar form
		polar = conv2polar(shifted)
		#print('Rad : ', polar[0])
		#print('Theta : ', polar[1])
		theta = polar[1]
		cv2.line(frame, (int(control_shape[1]/2), int(control_shape[0]/2)), (tracked[0], tracked[1]), (255,255,255), 1)
		#Find the polar sectors
		eye_sector = sector(theta)
		#Check locations
		if eye_sector == marker_sector:
			if delay == 0: delay = time.time()
			check = time.time() - delay
			print('Watching')trigger
			if check > 1:
				#Flip Switch
				if trigger == 0: #Off before
					trigger = 1
					print('Turning On')
				elif trigger == 1: #On before
					trigger = 0
					print('Turning Off')
				arduino.write(struct.pack('>B', trigger))
				delay = 0
		else: delay = 0
	detected = cv2.resize(detected,(950,980),interpolation = cv2.INTER_AREA)
	cv2.imshow('Control', detected)
	cv2.imshow("Pupil Tracking", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# show the output image with the face detections + facial landmarks

cv2.waitKey(0)
