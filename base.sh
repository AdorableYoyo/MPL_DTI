#!/bin/bash
cd /raid/home/yoyowu/MPL_DTI
device=4
exp_name=0614_1
global_step=30000
eval_at=100
batch_size=128
runseed=14
frozen=whole
python baseline.py \
--device ${device} --exp_name ${exp_name} --global_step ${global_step} \
--batch_size ${batch_size} --runseed ${runseed} --frozen ${frozen} \
--eval_at ${eval_at} --filename base_${frozen}_${exp_name}_step_${global_step}_bs_${batch_size}_run_${runseed} \
> logs/baseline_${frozen}_${exp_name}_step_${global_step}_bs_${batch_size}_run_${runseed}_output.log 2>&1 & 
echo "done ${exp_name}"

