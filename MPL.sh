#!/bin/bash
cd /raid/home/yoyowu/MPL_DTI
device=7
exp_name=0603_0
global_step=30000
eval_at=100
batch_size=64
runseed=58
mu=4
python main.py \
--device ${device} --exp_name ${exp_name} --global_step ${global_step} \
--batch_size ${batch_size} --runseed ${runseed} --mu ${mu} \
--eval_at ${eval_at} --filename MPL_${exp_name}_step_${global_step}_mu_${mu}_bs_${batch_size}_seed_${runseed} \
> logs/MPL_${exp_name}_step_${global_step}_bs_${batch_size}_mu_${mu}_seed_${runseed}_output.log 2>&1 & 
echo "done ${exp_name}"

