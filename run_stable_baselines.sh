#!/bin/sh

for env in 'HalfCheetah-v2' 'Ant-v3' 'Walker2d-v3' 'Hopper-v3' 'Pendulum-v1'
do
    python3 train_stable_baselines.py --env_name $env --algo td3 --device cuda:0 &
    python3 train_stable_baselines.py --env_name $env --algo td3 --device cuda:0 &
    python3 train_stable_baselines.py --env_name $env --algo td3 --device cuda:0 &
    python3 train_stable_baselines.py --env_name $env --algo td3 --device cuda:1 &
    python3 train_stable_baselines.py --env_name $env --algo td3 --device cuda:1 &
    wait
done

