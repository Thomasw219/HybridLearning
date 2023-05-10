import gym
import d4rl # Import required to register environments
import os
import numpy as np
import argparse

from stable_baselines3 import TD3, PPO, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

from sac_lib import EpisodeLengthWrapper
from comet_ml import Experiment

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
parser.add_argument('--algo', type=str, default='td3')

args = parser.parse_args()

experiment = Experiment(
    api_key = "e1Xmlzbz1cCLgwe0G8m7G58ns",
    project_name = args.algo,
    workspace="thomasw219",
)
experiment.add_tag(args.env_name)
experiment.log_parameters(vars(args))

TOTAL_STEPS = 1000000
EVAL_EVERY = 1000
N_EVAL_EPISODES = 10
EPISODE_LENGTH = 100

# Create the environment
env = EpisodeLengthWrapper(gym.make(args.env_name), EPISODE_LENGTH)
env = Monitor(env)

if args.algo == 'td3':
    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        device=args.device,
    )
elif args.algo == 'ppo':
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=args.device,
    )
elif args.algo == 'ddpg':
    model = DDPG(
        "MlpPolicy",
        env,
        verbose=1,
        device=args.device,
    )
else:
    raise NotImplementedError(f"Algo {args.algo} not implemented")

eval = lambda model: evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)

for step in range(np.ceil(TOTAL_STEPS / EVAL_EVERY).astype(int)):
    print(f"Step: {step}")
    model.learn(
        EVAL_EVERY,
        progress_bar=True,
        )
    mean, _ = eval(model)
    episode = (step + 1) * EVAL_EVERY / EPISODE_LENGTH
    experiment.log_metric("eval/reward_mean", mean, step=episode)
