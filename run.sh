#!/bin/sh

# for env in 'HalfCheetah-v2' 'Ant-v3' 'Walker2d-v3' 'Hopper-v3' 'Pendulum-v2'
# for env in 'HalfCheetah-v2'
# do
#     python3 train_hlt.py --seed 0 --env $env --method sac__ --device cuda:0 &
#     python3 train_hlt.py --seed 0 --env $env --method mpc_stoch --device cuda:0 &
#     python3 train_hlt.py --seed 0 --env $env --method hlt_stoch --device cuda:1 &
#     wait
#     # for i in $(seq 13 100 950)
#     # do
#     # for method in 'sac__' 'hlt_stoch' 'mpc_stoch'
#     # do
#     #     echo $env $i $method
#     #     python3 train_hlt.py --seed 0 --env $env --method $method
#     # done
#     # done
# done

for method in 'hlt_stoch' #'sac__' 'hlt_stoch' 'mpc_stoch'
do
    for env in 'Walker2d-v3'
    do
        python3 train_hlt.py --seed 0 --env $env --method $method --device cuda:0 --alpha .2 --no_entropy_backup --q_layer_norm &
        python3 train_hlt.py --seed 1 --env $env --method $method --device cuda:1 --alpha .2 --no_entropy_backup --q_layer_norm &
        python3 train_hlt.py --seed 2 --env $env --method $method --device cuda:2 --alpha .2 --no_entropy_backup --q_layer_norm &
        python3 train_hlt.py --seed 3 --env $env --method $method --device cuda:3 --alpha .2 --no_entropy_backup --q_layer_norm &
        python3 train_hlt.py --seed 4 --env $env --method $method --device cuda:0 --alpha .2 --no_entropy_backup --q_layer_norm &
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
done