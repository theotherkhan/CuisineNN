#!/bin/bash 

if  pip3 show simplejson  > /dev/null
then
    :
else
    echo "json installing..."
    pip3 install simplejson
    echo "json installed"
fi

if  pip3 show bitarray  > /dev/null
then
    :
else
    echo "bitarray installing..."
    pip3 install bitarray
    echo "bitarray installed"
fi

python3 machine_learning.py