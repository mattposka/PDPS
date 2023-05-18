#!/bin/bash

 

# PATHNAME=`pwd`

PATHNAME=/home/pi/rapa

echo $PATHNAME

 

DIRNAME="images"

DS=$(date +"%Y-%m-%d_%H00")

 

#echo $DIRNAME

echo $DS

 

if ! [ -d $PATHNAME/$DIRNAME ]

then

    echo "The $DIRNAME folder does not exist."

    echo "The array may be stopped already."

    exit 2

fi

 

if [ -d $PATHNAME/$DIRNAME.$DS ]

then

    echo "$DIRNAME.$DS exists."

    echo "The array may be stopped already."

    exit 4

else

    mv $PATHNAME/$DIRNAME $PATHNAME/$DIRNAME.$DS

    cp $PATHNAME/bin/image.off $PATHNAME/bin/image.sh

 

fi
