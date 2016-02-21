import numpy as np
import sys
import os
import subprocess
import time
import cv2

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

def saveVideoFrames(videoFile):
	cap = cv2.VideoCapture(videoFile)
	fgbg = cv2.BackgroundSubtractorMOG()

	directory = "randomMovement" #str(time.time())

	if not os.path.exists(directory):
		os.makedirs(directory)

	frameIndex = 6630

	startTime = time.time()

	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(gray, (32, 32))

		relativePath = directory
		framePath = 'frame' + str(frameIndex) + '.jpg'
		fullPath = os.path.join(relativePath, framePath)

		cv2.imwrite(fullPath, img)

		frameIndex = frameIndex + 1

		# Display the resulting frame
		cv2.imshow('frame', img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()