#!/bin/bash 

if  pip show simplejson  > /dev/null
then
    :
else
    echo "json installing..."
    pip install simplejson
    echo "json installed"
fi

if  pip show bitarray  > /dev/null
then
    :
else
    echo "bitarray installing..."
    pip install bitarray
    echo "bitarray installed"
fi

python2 driver.py ingredients.json training.json