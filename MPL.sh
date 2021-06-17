#!/bin/bash
cd /raid/home/yoyowu/MPL_DTI
device=5
exp_name=0613_7
global_step=60000
eval_at=100
batch_size=64
runseed=6
mu=4
label_smoothing=0.15
temperature=1.0
ratio_low=0.8
ratio_high=1.4
threshold=0.7
python main_filters.py \
--device ${device} --exp_name ${exp_name} --global_step ${global_step} --temperature ${temperature} \
--ratio_low ${ratio_low} --ratio_high ${ratio_high} --threshold ${threshold} \
--batch_size ${batch_size} --runseed ${runseed} --mu ${mu} --label_smoothing ${label_smoothing} \
--eval_at ${eval_at} \
--filename MPL_blc_conf${exp_name}_${global_step}_mu_${mu}_${batch_size}_run_${runseed}_smt_${label_smoothing}_tem_${temperature}_blc${ratio_low}_${ratio_high}_conf${threshold} \
> logs/MPL_blc_conf${exp_name}_${global_step}_${batch_size}_mu_${mu}_run_${runseed}_smt_${label_smoothing}_tem_${temperature}_blc${ratio_low}${ratio_high}_conf${threshold}_output.log 2>&1 & 
echo "done ${exp_name}"

