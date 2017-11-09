#!/bin/bash 

if  pip3 show json  > /dev/null
then
    :
else
    echo "json installing..."
    pip3 install json > /dev/null
    echo "json installed"
fi

python3 jsonReader.py