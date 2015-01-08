#!/bin/bash
echo 'Argument $0 = ' $0
echo 'Argument $1 = ' $1
echo 'Argument $2 = ' $2
echo 'Argument $3 = ' $3
for ((i = $2; i <= $3; i++));
do
    echo "python ../optskills/main.py $1 $i"
    python ../optskills/main.py $1 $i
done
