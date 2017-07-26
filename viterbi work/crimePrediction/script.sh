#!/bin/bash
# script to test model for predicting crime using past data

#echo 'Hope you have given arguments in following order : rows, cols, time_period, lookback'

#declare -a arrRow=(12)
#declare -a arrRow=(8)
row=12
col=22

#declare -a arrCol=(22)
#declare -a arrCol=(8)

declare -a arrTimePeriod=(24)
#declare -a arrTimePeriod=(24)

declare -a arrLookBack=(2)
#declare -a arrLookBack=(7)


for k in "${arrTimePeriod[@]}"
do
	for l in "${arrLookBack[@]}"
	do
		python makeGrid.py $row $col
		python dividebyTime.py $row $col $k
		python crimePrediction.py $row $col $k $l				
	done
done


:'
python makeGrid.py $1 $2
python dividebyTime.py $1 $2 $3
python crimePrediction.py $1 $2 $3 $4
'