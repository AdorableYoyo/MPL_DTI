#!/bin/bash
cd /raid/home/yoyowu/MPL_DTI
device=6
exp_name=0614_10
global_step=60000
eval_at=100
batch_size=64
runseed=0
mu=4
label_smoothing=0.1
temperature=1.0
threshold=0.7
python main_conf.py \
--device ${device} --exp_name ${exp_name} --global_step ${global_step} --temperature ${temperature} \
--threshold ${threshold} \
--batch_size ${batch_size} --runseed ${runseed} --mu ${mu} --label_smoothing ${label_smoothing} \
--eval_at ${eval_at} \
--filename MPL_conf${exp_name}_${global_step}_mu_${mu}_${batch_size}_run_${runseed}_smt_${label_smoothing}_tem_${temperature}_conf${threshold} \
> logs/MPL_conf${exp_name}_${global_step}_${batch_size}_mu_${mu}_run_${runseed}_smt_${label_smoothing}_tem_${temperature}_conf${threshold}_output.log 2>&1 & 
echo "done ${exp_name}"

