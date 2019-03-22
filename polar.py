#Electronics Club IITG - TechEvince 2019
#Guining Pertin(Mentor), Aadi Gupta, Aadarsh Khandelwal, Shridam Mahajan

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
	Input : img : the cropped eye
	Output :	  returns the pupil location along x
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
	Input : img : the cropped eye
	Output : 	  returns the pupil location along y
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
	"""
	Credits to Tobias Roeddiger - https://github.com/TobiasRoeddiger/
	A function to find the pupil location along y axis by finding minimum intensities
	Input : image : face webcam input
			left,...upper_left : the detected landmarks
	Output : eye : the cropped eye
			 pupil_x, pupil_y : the pupil location
	"""
	#Find the maximum and mimimum bound along y axis for eye
	lower_bound = max([left[1], right[1], bottom_left[1], bottom_right[1], upper_left[1], upper_right[1]])
	upper_bound = min([left[1], right[1], upper_left[1], upper_right[1], bottom_left[1], bottom_right[1]])
	#Cropped eye
	eye = image[upper_bound-3:lower_bound+3, left[0]-3:right[0]+3]
	#Determine the pupil location
	pupil_x = min_intensity_x(eye)
	pupil_y = min_intensity_y(eye)
	#Draw cross-lines to show the pupil location
	cv2.line(eye,(pupil_x,0),(pupil_x,len(eye)),(0,255,0), 1)
	cv2.line(eye,(0,pupil_y),(len(eye[0]),pupil_y),(0,255,0), 1)
	#Draw cross-lines to show center of the eye
	cv2.line(image,(int((bottom_left[0] + bottom_right[0]) / 2), lower_bound), (int((upper_left[0] + upper_right[0]) / 2), upper_bound),(0,0,255), 1)
	cv2.line(image,(left[0], left[1]), (right[0], right[1]),(0,0,255), 1)
	image[upper_bound-3:lower_bound+3, left[0]-3:right[0]+3] = eye
	return eye, pupil_x, pupil_y

def mapper(value, leftMin, leftMax, rightMin, rightMax):
	"""
	A function to map a value from present range to a required range
	Input : value : current value
			leftMin, leftMax : present range
			rightMin, rightMax : required range
	Output : mapped value
	"""
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float((value - leftMin) / leftSpan)
    return int(rightMin + (valueScaled * rightSpan))

def conv2polar(coor):
	"""
	A function to convert from cartesian to polar coordinates
	Input : coor : a tuple (x,y) with the coordinates
	Output : polar : a tuple (r, theta) with polar coordinates
	"""
	r = int(np.sqrt(coor[0]*coor[0] + coor[1]*coor[1]))
	theta = int(np.arctan2(coor[1], coor[0])*180/3.14)
	polar = (r, theta)
	return polar

def sector(theta):
	"""
	A function to determine the current sector, given the angle
	Input : theta : angle wrt center
	Output : sec : current sector
	"""
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
	#Get face_cam and user_cam input
	_, image = face_cam.read()
	_, frame = user_cam.read()
	image = np.rot90(image, 2)	#our camera was inverted
	image = imutils.resize(image, width=500)	#resizing face_cam input
	#Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#Set variables for marker and eye sectors
	marker_sector = 0
	eye_sector = 0
	#Get the user_cam input shape
	control_shape = frame.shape

	#Marker detection
	#Detect ArUco marker
	corners, ids, rejected = aruco.detectMarkers(gray2, aruco_dict,parameters = params)
	detected = aruco.drawDetectedMarkers(frame, corners)
	#If at least one marker is detected
	if np.all(ids != None):
		#Get marker center
		marker_x = int((corners[0][0][0][0] + corners[0][0][2][0])/2)
		marker_y = int((corners[0][0][0][1] + corners[0][0][2][1])/2)
		cv2.circle(detected, (marker_x, marker_y), 5, (255,0,0), -1)
		#Convert to polar coordinates after oprign shift
		marker_shift = (marker_x - int(control_shape[1]/2), marker_y - int(control_shape[0]/2))
		marker_polar = conv2polar(marker_shift)
		marker_sector = sector(marker_polar[1])

	#Face and landmark detection
	#Detect face using HOG+SVM based detector
	detected_faces = detector(gray, 1)
	#Loop over each face detected
	for (i, rect) in enumerate(detected_faces):
		#Determine face landmarks
		shape = predictor(gray, rect)
		#Conver the landmarks to numpy array
		shape = imutils.face_utils.shape_to_np(shape)
		count = 1
		#Get the extracted eye
		eye_out, centerx, centery = extract_eye(image, shape[36], shape[41], shape[40], shape[39], shape[38], shape[37])
		#Resize the extracted eye and draw the pupil center
		right_eye = imutils.resize(eye_out, width=100, height=50)
		cv2.circle(image, (centerx, centery), 10, (255,255,255), 1)
		#Draw eye landmarks - 37 to 42
		for (x, y) in shape:
			if count > 36 and count < 43:
					cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
			count += 1
		#Show the right eye on the top-left corner
		image[0:len(right_eye),0:len(right_eye[0])] = right_eye

		#Mapping
		eye_shape = right_eye.shape
		#Get the mapped coordinates
		controlx = int(control_shape[1]/2) + int(2.8*mapper(centerx,0, eye_shape[1]-45, -1*int(control_shape[1]/2), int(control_shape[1]/2)))
		controly = int(control_shape[0]/2) + int(2*mapper(centery, 0, 8, -1*int(control_shape[0]/2)-50, int(control_shape[1]/2)))+20
		tracked = [control_shape[0]-controlx, controly]
		#Limits
		if (tracked[0] < 0): tracked[0] = 0
		elif (tracked[0] > control_shape[1]): tracked[0] = control_shape[1]
		if (tracked[1] < 0): tracked[1] = 1
		elif (tracked[1] > control_shape[0]): tracked[1] = control_shape[0]

		#Control signal
		#Convert to tuple
		tracked = (tracked[0], tracked[1])
		#Display location on the user_cam input
		cv2.circle(frame,tracked, 10, (255,255,255), -1)
		#Origin shift
		track_shift = (tracked[0]-(control_shape[1]/2), tracked[1]-(control_shape[0]/2))
		#Convert to polar form
		polar = conv2polar(track_shift)
		theta = polar[1]
		#Draw the line joining mapped point to the center
		cv2.line(frame, (int(control_shape[1]/2), int(control_shape[0]/2)), (tracked[0], tracked[1]), (255,255,255), 1)
		#Find the polar sectors
		eye_sector = sector(theta)
		#Check locations
		if eye_sector == marker_sector:
			#Determine time elapsed
			if delay == 0: delay = time.time()
			check = time.time() - delay
			print('Watching')
			if check > 1:
				#Flip Switch
				if trigger == 0: #Off before
					trigger = 1
					print('Turning On')
				elif trigger == 1: #On before
					trigger = 0
					print('Turning Off')
				arduino.write(struct.pack('>B', trigger))
				#Reset time elapsed
				delay = 0
		#Reset time elapsed
		else: delay = 0

	#Resize the image
	#detected = cv2.resize(detected,(950,980),interpolation = cv2.INTER_AREA)
	#Show the final outputs
	cv2.imshow('Control', detected)
	cv2.imshow("Pupil Tracking", image)
	#Wait until q is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
