#!/bin/bash
cd /raid/home/yoyowu/MPL_DTI
device=2
exp_name=0525_0
global_step=30000
eval_at=100
batch_size=128
runseed=49
python baseline.py \
--device ${device} --exp_name ${exp_name} --global_step ${global_step} \
--batch_size ${batch_size} --runseed ${runseed} \
--eval_at ${eval_at} --filename ${exp_name}_step_${global_step}_bs_${batch_size}_seed_${runseed} \
> logs/baseline_${exp_name}_step_${global_step}_bs_${batch_size}_seed_${runseed}_output.log 2>&1 & 
echo "done ${exp_name}"

