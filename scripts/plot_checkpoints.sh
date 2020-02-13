#!/usr/bin/env/python
checkpointdir=$1
seq_start=$2
seq_step=$3
seq_end=$4
dim0=$5
dim1=$6
plot_one_dim=$7
use_qpos=$8
#checkpointdir=demo/seed_1_swimmer_alpha_0.1
#for i in $(seq 0  10 60)
for i in $(seq ${seq_start}  ${seq_step} ${seq_end})
do
	path=${checkpointdir}/itr_$i.pkl
	#echo $path
	python3 /home/akshay/mtom_research/sac/scripts/plot_traces.py $path --dim_0 $dim0 --dim_1 $dim1 --plot-one-dim $plot_one_dim --use_qpos $use_qpos
done
