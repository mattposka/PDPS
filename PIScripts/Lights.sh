#!/bin/bash

 

#Image taking script for Mark Guiltinan's cacao pathogen assay.

#Included are lines to turn the backlighing on and off, assuming

# using wiringpi as outlined here:

# http://www.instructables.com/id/Controlling-Any-Device-Using-a-Raspberry-Pi-and-a/

# These lines are commented, simply uncomment to activate them.

 

##Turn lights on (pin 0, physical pin 11)

gpio mode 0 out

gpio write 0 0

## Pause for LED stabilization

sleep  180

 

##Take the picture

#pinum=`cat /home/pi/rapa_config.txt`

#DATE=$(date +"%Y-%m-%d_%H%M")

#raspistill --ISO 100 --sharpness 25 --exposure backlight --awb fluorescent --width 3280 --height 2464 -e png -o /home/pi/rapa/images/rapa_cam_"$pinum"_"$DATE".png

# Same image parameters, but greyscale image

#

#raspistill --ISO 100 --sharpness 25 --exposure backlight --awb fluorescent --width 3280 --height 2464 -cfx 128:128 -e png -o /home/pi/rapa/images/rapa_camBW_"$pinum"_"$DATE".png

 

##Turn lights off

gpio write 0 1
