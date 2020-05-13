import numpy as np
import random
import pickle
from datetime import datetime

import sys
import os
import yaml

# local imports
import envs
import gym
from gym import wrappers

import torch
from sac_lib import SoftActorCritic
from sac_lib import PolicyNetwork
from sac_lib import ReplayBuffer
from sac_lib import NormalizedActions
from hlt_lib import StochPolicyWrapper, DetPolicyWrapper
from model import ModelOptimizer, Model, SARSAReplayBuffer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env',   type=str,   default='InvertedPendulumBulletEnv')
parser.add_argument('--method', type=str, default='hlt_stoch')
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=True)
parser.add_argument('--record', dest='record', action='store_true')
parser.add_argument('--no-record', dest='record', action='store_false')
parser.set_defaults(record=False)

args = parser.parse_args()


if __name__ == '__main__':

    config_path = './config/' + args.method + '.yaml'
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = config_dict['default']
        if args.env in list(config_dict.keys()):
            config.update(config_dict[args.env])
        else:
            raise ValueError('env not found config file')

    env_name = args.env
    try:
        env = NormalizedActions(envs.env_list[env_name](render=args.render))
    except TypeError as err:
        print('no argument render,  assumping env.render will just work')
        env = NormalizedActions(envs.env_list[env_name]())
    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'

    if args.record:
        env = gym.wrappers.Monitor(env, 'recording', force=True)
    env.reset()

    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 128

    device ='cpu'
    if torch.cuda.is_available():
        device  = 'cuda:0'
        print('Using GPU Accel')

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
    model = Model(state_dim, action_dim, def_layers=[200]).to(device)

    state_dict_path = './data/' + config['method'] + '/' + env_name + '/seed_{}/'.format(args.seed)

    policy_net.load_state_dict(torch.load(state_dict_path+'policy_final.pt', map_location=device))
    model.load_state_dict(torch.load(state_dict_path+'model_final.pt', map_location=device))

    if config['method'] == 'hlt_stoch':
        hybrid_policy = StochPolicyWrapper(model, policy_net,
                                samples=config['trajectory_samples'],
                                t_H=config['horizon'],
                                lam=config['lam'])
    elif config['method'] == 'hlt_deter':
        hybrid_policy = DetPolicyWrapper(model, policy_net,
                                    T=config['horizon'],
                                    lr=config['planner_lr'])
    else:
        ValueError('method not found in config')

    max_frames  = config['max_frames']
    max_steps   = config['max_steps']
    frame_skip  = config['frame_skip']

    frame_idx   = 0
    rewards     = []

    ep_num = 0
    state = env.reset()
    hybrid_policy.reset()

    episode_reward = 0
    done = False
    for step in range(max_steps):
        action = hybrid_policy(state)
        for _ in range(frame_skip):
            state, reward, done, _ = env.step(action.copy())
        episode_reward += reward
        frame_idx += 1

        if args.render:
            env.render("human")

        if args.done_util:
            if done:
                break
    rewards.append([frame_idx, episode_reward])
    ep_num += 1
    env.close()
