#!/bin/bash

 

# PATHNAME=`pwd`

PATHNAME=/home/pi/rapa

echo $PATHNAME

 

DIRNAME="images"

 

if ! [ -d $PATHNAME/$DIRNAME ]

then

    echo "Creating image directory and changing script."

    mkdir $PATHNAME/$DIRNAME

    cp $PATHNAME/bin/image.on $PATHNAME/bin/image.sh

else

    echo "The $DIRNAME folder already exists"

    echo "Please see if the array is already running"

    exit 2

 

fi

 

echo "The array has been started."

exit 0
