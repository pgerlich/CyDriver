# import RPi.GPIO as GPIO
# from time import sleep
# import sys, tty, termios
# import picamera
# import pygame
import socket
import sys

# pygame.init()

# #cam = picamera.PiCamera()
# #cam.vflip = True
# picNum = 0

# filming = False
# front = False
# back = False
# left = False
# right = False
 
# GPIO.setmode(GPIO.BOARD)
 
# Motor1A = 16
# Motor1B = 18
# Motor1E = 22
 
# Motor2A = 21
# Motor2B = 19
# Motor2E = 23
 
# GPIO.setup(Motor1A,GPIO.OUT)
# GPIO.setup(Motor1B,GPIO.OUT)
# GPIO.setup(Motor1E,GPIO.OUT)
# Enable1 = GPIO.PWM(Motor1E, 100)
# Enable1.start(0)
 
# GPIO.setup(Motor2A,GPIO.OUT)
# GPIO.setup(Motor2B,GPIO.OUT)
# GPIO.setup(Motor2E,GPIO.OUT)
# Enable2 = GPIO.PWM(Motor2E, 100)
# Enable2.start(0)

# print "ready to drive"
# def forward()       :
#     print "Going forwards"
    
#     #GPIO.output(Motor1A,GPIO.LOW)
#     #GPIO.output(Motor1B,GPIO.HIGH)
     
#     GPIO.output(Motor1A,GPIO.HIGH)
#     GPIO.output(Motor1B,GPIO.LOW)
#     Enable1.ChangeDutyCycle(100)    

# def backwards():
#     print "Going backwards"

#     #GPIO.output(Motor1A,GPIO.LOW)
#     #GPIO.output(Motor1B,GPIO.HIGH) 
#     GPIO.output(Motor1A,GPIO.LOW)
#     GPIO.output(Motor1B,GPIO.HIGH)
#     Enable1.ChangeDutyCycle(100)

# def straight():
#     Enable1.ChangeDutyCycle(100)
#     Enable2.ChangeDutyCycle(100)    
    
# def rightTurn():
#     print "going right"
#     GPIO.output(Motor2A, GPIO.HIGH)
#     GPIO.output(Motor2B, GPIO.LOW)
#     Enable1.ChangeDutyCycle(100)
    
# def leftTurn():
#     print "going left"  
#     GPIO.output(Motor2A, GPIO.LOW)
#     GPIO.output(Motor2B, GPIO.HIGH)
#     Enable1.ChangeDutyCycle(100)
    
# def halt():
#     print "stopping"
#     front = False
#     left = False
#     back = False
#     right = False
#     GPIO.output(Motor1A,GPIO.LOW)
#     GPIO.output(Motor1B,GPIO.LOW)
#     Enable1.ChangeDutyCycle(0)
    
#     GPIO.output(Motor2A,GPIO.LOW)
#     GPIO.output(Motor2B,GPIO.LOW)
#     Enable2.ChangeDutyCycle(0)
        
# def straighten():
#     if (front or back):
#         straight()  
    

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('localhost', 10000)
print >>sys.stderr, 'starting up on %s port %s' % server_address
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    print >>sys.stderr, 'waiting for a connection'
    connection, client_address = sock.accept()

    try:
        print >>sys.stderr, 'connection from', client_address

        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(16)

            print data

            # if "FWD" in data:
            # 	front = True
            #     back = False
            #     straight()
            #     forward()
            # 	print "Moving FWD"
            # elif "LEFT" in data:
            # 	right = False
            #     left = True
            #     leftTurn()
            # 	print "Moving Left"
            # elif "RIGHT" in data:
            #     right = True
            #     left = False
            #     rightTurn()
            # 	print "Moving Right"


            if not data:
                break
            
    finally:
        # Clean up the connection
        connection.close()