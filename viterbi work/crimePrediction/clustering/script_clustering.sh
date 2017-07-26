#!/bin/bash
#script to test model while varying clustering parameters

rows=10
cols=10
declare -a eps=(0.01 0.05 0.10 0.15)
declare -a minSamples=(1 5 10 15)
timePeriod=168
lookback=2
#declare -a countThreshold=(0 100 1000 5000 10000 15000 20000 40000)
#declare -a countThreshold=(5000)
method=1

for epsilon in "${eps[@]}"
do
	for minSample in "${minSamples[@]}"
	do
		python clustering.py $rows $cols $epsilon $minSample
		#python divideByTimeClustering.py $rows $cols $timePeriod
		python crimePredictionKFold.py $rows $cols $timePeriod $lookback $epsilon $method $minSample >> 'results_kfold.txt'
	done
done

#for threshold in "${countThreshold[@]}"
#do
#	echo $threshold	
#	python gridClustering.py $rows $cols $timePeriod $lookback $threshold
#done
