#CyDriver
CyDriver is a Node, RaspberryPi, and Deep learning combination project.

#The beginning
We disassembled two stock RC cars from Wal-Mart. We then stripped the inner componenets and hooked up a Raspberry Pi to control the servos.

#Node.js, WIFI & GPIO
From there we turned the Raspberry Pis into routers broadcasting an SSID so that they can be remotely controlled through wifi. With the network connection complete, we setup Node.js servers on the pis so that a public web interface could be accessed to grant control of the vehicle. The vehicle is controlled with websockets and _ javascript library for interfacing directly with the GPIO pins on the Pi.

#A Step Beyond
After we completed this, we figured we would spice things up a bit. As the Raspberry Pi is being manually controlled, it is saving the video feed and logging which direction it is going. With this data, we are able to train a Deep Convolutional Neural Network to drive a seperate car on a seperate track - with hopefully similar results.

#Deep Learning
With the data collected, and a network trained - we are able to control another vehicle. As no network will attain 100% accuracy, we use a sort of rolling average algorithm to choose the most confident direction over the last _ frames so that the car will drive smoothly - as opposed to whimsically changing directions. 

#Next Steps
What's next? I don't know.
