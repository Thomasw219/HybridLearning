#!/bin/sh

for env in 'HalfCheetah-v2' 'Ant-v3' 'Walker2d-v3' 'Hopper-v3' 'Pendulum-v2'
do
    python3 train_hlt.py --seed 0 --env $env --method sac__ --device cuda:0 &
    python3 train_hlt.py --seed 0 --env $env --method mpc_stoch --device cuda:0 &
    python3 train_hlt.py --seed 0 --env $env --method hlt_stoch --device cuda:1 &
    wait
    # for i in $(seq 13 100 950)
    # do
    # for method in 'sac__' 'hlt_stoch' 'mpc_stoch'
    # do
    #     echo $env $i $method
    #     python3 train_hlt.py --seed 0 --env $env --method $method
    # done
    # done
done
