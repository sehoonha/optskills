#!/bin/bash
echo 'Argument $0 = ' $0
echo 'Argument $1 = ' $1
for i in {5..10}
do
    echo "python ../optskills/main.py $1 $i"
    python ../optskills/main.py $1 $i
done
