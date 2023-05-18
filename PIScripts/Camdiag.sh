#!/bin/bash

 

#Image taking script for Mark Guiltinan's cacao pathogen assay.

#Included are lines to turn the backlighing on and off, assuming

# using wiringpi as outlined here:

# http://www.instructables.com/id/Controlling-Any-Device-Using-a-Raspberry-Pi-and-a/

# These lines are commented, simply uncomment to activate them.

 

PATHNAME=/home/pi/rapa

echo $PATHNAME

DIRNAME="diag"

FNAME="testcam"

 

if ! [ -e $PATHNAME/$DIRNAME/$FNAME ]

then

#    echo "Do nothing"

    exit 2

else

#    echo "Do something"

    ##Turn lights on (pin 0, physical pin 11)

    gpio mode 0 out

    gpio write 0 0

    ### Pause for LED stabilization

    sleep 5

 

    ##Take the picture

    pinum=`cat /home/pi/rapa_config.txt`

 

    raspistill --ISO 100 --sharpness 25 --exposure backlight --awb fluorescent --width 3280 --height 2464 -e png -o /home/pi/rapa/diag/rapa_cam_"$pinum".png

 

    ##Turn lights off

    gpio write 0 1

 

 

    rm $PATHNAME/$DIRNAME/$FNAME

    exit 0

fi

 

echo $?

 

exit 0
