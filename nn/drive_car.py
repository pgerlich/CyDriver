import numpy as np
import sys
from collections import Counter
import os
import RPi.GPIO as GPIO
from time import sleep
import sys, tty, termios
import picamera
import pygame
import subprocess
import time
import cv2

#Initialize connections
cam = picamera.PiCamera()
cam.vflip = True
picNum = 0

filming = False
front = False
back = False
left = False
right = False
 
GPIO.setmode(GPIO.BOARD)
 
Motor1A = 16
Motor1B = 18
Motor1E = 22
 
Motor2A = 21
Motor2B = 19
Motor2E = 23
 
GPIO.setup(Motor1A,GPIO.OUT)
GPIO.setup(Motor1B,GPIO.OUT)
GPIO.setup(Motor1E,GPIO.OUT)
Enable1 = GPIO.PWM(Motor1E, 100)
Enable1.start(0)
 
GPIO.setup(Motor2A,GPIO.OUT)
GPIO.setup(Motor2B,GPIO.OUT)
GPIO.setup(Motor2E,GPIO.OUT)
Enable2 = GPIO.PWM(Motor2E, 100)
Enable2.start(0)

def main():
	try:
		videoInputFile = str(sys.argv[1])
	except IndexError:
		print "Expecting video input file as argument"

	print "Converting video to MP4"

	videoOutputFile = videoInputFile.split('.')[0] + ".mp4"

	#Use sub-process to rip inputVideo to outputWav
	command = "ffmpeg -i " + videoInputFile + " -c:v copy " + videoOutputFile
	subprocess.call(command, shell=True)

	print "Conversion complete."

	print "Ripping frames"

	saveVideoFrames(videoOutputFile)

	print "Rip complete."

def driveCarFromImage():
	cap = cv2.VideoCapture(0)

	lastMoves = numpy.zeros(38)
	moveIndex = 0

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(gray, (100, 100))

		#Classify image
		curLabel = 0 #TODO Classify image from NN code

		#Append moves to do rolling average
		lastMoves[moveIndex % 38] = curLabel

		highMode = Counter(lastMoves).most_common(1) #Get the most occuring value in the last 38 predictions

		move(highMode)

		moveIndex = moveIndex + 1

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def move(dir):
	if dir == 0:
		forward()
	if dir == 1:
		leftTurn()
	if dir == 2:
		rightTurn()

def forward():
	GPIO.output(Motor1A,GPIO.HIGH)
	GPIO.output(Motor1B,GPIO.LOW)
	Enable1.ChangeDutyCycle(100)	
	
def rightTurn():
	GPIO.output(Motor2A, GPIO.HIGH)
	GPIO.output(Motor2B, GPIO.LOW)
	Enable1.ChangeDutyCycle(100)
	
def leftTurn():
	GPIO.output(Motor2A, GPIO.LOW)
	GPIO.output(Motor2B, GPIO.HIGH)
	Enable1.ChangeDutyCycle(100)
	

if __name__ == '__main__':
	main()