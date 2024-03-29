import numpy as np
import random
import pickle
from datetime import datetime, timezone, timedelta
import gym
from comet_ml import Experiment
from torch.utils.tensorboard import SummaryWriter

import sys
import os
import yaml

import torch
# from sac_lib import SoftActorCritic
# from sac_lib import PolicyNetwork
# from sac_lib import ReplayBuffer
from sac_lib.alternate_sac import SAC, ReplayMemory
from sac_lib import NormalizedActions, EpisodeLengthWrapper
from hlt_lib import StochPolicyWrapper, DetPolicyWrapper
from model import ModelOptimizer, Model, SARSAReplayBuffer
from mpc_lib import ModelBasedDeterControl, PathIntegral

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--env',   type=str,   default='HalfCheetah-v2')
parser.add_argument('--method', type=str, default='hlt_stoch')
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--done_util', dest='done_util', action='store_true')
parser.add_argument('--no_done_util', dest='done_util', action='store_false')
parser.set_defaults(done_util=True)
parser.add_argument('--log', dest='log', action='store_true')
parser.add_argument('--no-log', dest='log', action='store_false')
parser.set_defaults(log=True)
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no_render', dest='render', action='store_false')
parser.set_defaults(render=False)

parser.add_argument('--policy_type', default="tanh_gaussian")
parser.add_argument('--std_scale',type=float,default=1.0)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--automatic_entropy_tuning', action='store_true')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=1000001)
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument('--updates_per_step', type=int, default=1)
parser.add_argument('--start_steps', type=int, default=10000)
parser.add_argument('--target_update_interval', type=int, default=1)
parser.add_argument('--eval_freq',type=int,default=10)
parser.add_argument('--deterministic_eval',action='store_true')
parser.add_argument('--n_eval_episodes',type=int,default=10)
parser.add_argument('--replay_size', type=int, default=1000000)
parser.add_argument('--q_ensemble_members', type=int, default=2)
parser.add_argument('--q_subset_size',type=int,default=2)
parser.add_argument('--q_layer_norm',action='store_true')
parser.add_argument('--no_entropy_backup',action='store_true')
parser.add_argument('--ep_len', type=int, default=None)
parser.add_argument('--save_gif',action='store_true')
parser.add_argument('--save_policy',action='store_true')
parser.add_argument('--no_termination',action='store_true')
parser.add_argument('--target_entropy',type=float,default=1.0,help='Sets the scale factor on the target entropy for automatic ent tuning')
parser.add_argument('--huber',action='store_true')
parser.add_argument('--sparse',action='store_true')
parser.add_argument('--vel_thresh',type=float,default=1.0)
parser.add_argument('--cpu',action='store_true')
parser.add_argument('--save_data',action='store_true')
parser.add_argument('--load_data',action='store_true')
parser.add_argument('--data_file_name',type=str,default=None)

args = parser.parse_args()
print(args)

class DMObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._max_episode_steps = env._max_episode_steps
        self.observation_space = env.observation_space['observations']

    def observation(self,obs):
        return obs['observations']

if __name__ == '__main__':
    base_method = args.method[:3]
    if args.method == 'sac__':
        config_path = './config/sac.yaml'
    elif args.method[4:] == 'deter':
        config_path = './config/hlt_deter.yaml'
    else:
        config_path = './config/hlt_stoch.yaml'

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = config_dict['default']
        if args.env in list(config_dict.keys()):
            config.update(config_dict[args.env])
        else:
            print('env not found config file')

    env_name = args.env
    env = gym.make(env_name, **({'environment_kwargs' : {'flat_observation':True}} if 'dm2gym' in env_name else {}))
    if 'dm2gym' in env_name:
        env = DMObsWrapper(env)
    env = EpisodeLengthWrapper(NormalizedActions(env), config['max_steps'])

    assert np.any(np.abs(env.action_space.low) <= 1.) and  np.any(np.abs(env.action_space.high) <= 1.), 'Action space not normalizd'
    if args.render:
        try:
            env.render() # needed for InvertedDoublePendulumBulletEnv
        except:
            print('render not needed')
    env.reset()

    env.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.log:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")
        dir_name = 'seed_{}/'.format(str(args.seed))
        path = './data/'  + args.method + '/' + env_name + '/' + dir_name
        if os.path.exists(path) == False:
            os.makedirs(path)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    hidden_dim = 256
    replay_buffer_size = 1000000
    replay_buffer = [None] # placeholder
    model_replay_buffer = [None] # placeholder

    device  = args.device

    timestring = datetime.now(tz=timezone(timedelta(hours=-4))).strftime("_%m-%d-%Y_%H-%M-%S") # EDT i think
    experiment = Experiment(
        api_key = "e1Xmlzbz1cCLgwe0G8m7G58ns",
        project_name = args.method if 'sac' not in args.method else 'sac',
        workspace="thomasw219",
    )
    experiment.add_tag(env_name)
    experiment.log_parameters(config)
    experiment.log_parameters(vars(args))
    logger = SummaryWriter(os.path.join('logs', f'{args.env}_{args.method}_seed_{args.seed}_{timestring}'))

    if base_method != 'sac':
        model = Model(state_dim, action_dim, def_layers=[200],AF=config['activation_fun']).to(device)
        model_replay_buffer = SARSAReplayBuffer(replay_buffer_size)
        model_optim = ModelOptimizer(model, model_replay_buffer, lr=config['model_lr'], device=device)

    if base_method != 'mpc':
        # policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim,AF=config['activation_fun']).to(device)
        # replay_buffer = ReplayBuffer(replay_buffer_size)
        # sac = SoftActorCritic(policy=policy_net,
        #                       state_dim=state_dim,
        #                       action_dim=action_dim,
        #                       replay_buffer=replay_buffer,
        #                       policy_lr=config['policy_lr'],
        #                       value_lr=config['value_lr'],
        #                       soft_q_lr=config['soft_q_lr'],
        #                       device=device,)
        replay_buffer = ReplayMemory(replay_buffer_size, args.seed)
        sac = SAC(
            state_dim=state_dim,
            action_space=env.action_space,
            args=args,
            )
        policy_net = sac.policy

    if args.method == 'hlt_stoch':
        hybrid_policy = StochPolicyWrapper(model, policy_net,
                                samples=config['trajectory_samples'],
                                t_H=config['horizon'],
                                lam=config['lam'],
                                device=device,)
    elif args.method == 'hlt_deter':
        hybrid_policy = DetPolicyWrapper(model, policy_net,
                                        T=config['horizon'],
                                        lr=config['planner_lr'],
                                        device=device,)
    elif args.method == 'mpc_stoch':
        planner = PathIntegral(model,
                               samples=config['trajectory_samples'],
                               t_H=config['horizon'],
                               lam=config['lam'],
                               device=device,)
    elif args.method == 'mpc_deter':
        planner = ModelBasedDeterControl(model, T=config['horizon'])
    elif base_method == 'sac':
        pass
    else:
        raise ValueError('method not found')

    max_frames  = config['max_frames']
    max_steps   = config['max_steps']
    frame_skip  = config['frame_skip']
    reward_scale = config['reward_scale']

    frame_idx   = 0
    rewards     = []
    batch_size  = 256

    ep_num = 0
    while (frame_idx < max_frames):
        state = env.reset()
        if base_method == 'sac':
            action = policy_net.get_action(state)
        elif base_method == 'mpc':
            planner.reset()
            action, _ = planner(state)
        else:
            hybrid_policy.reset()
            action, _ = hybrid_policy(state)

        episode_reward = 0
        done = False
        for step in range(max_steps):
            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action.copy())

            mask = 1 if (step + 1) == env._max_episode_length else float(not done)
            # mask = not mask
            if base_method == 'sac':
                next_action = policy_net.get_action(next_state)
                replay_buffer.push(state, action, reward, next_state, mask)
                if len(replay_buffer) > batch_size:
                    # sac.update(batch_size)
                    sac.update(replay_buffer, batch_size)
            elif base_method == 'mpc':
                next_action, _ = planner(next_state)
                model_replay_buffer.push(state, action, reward_scale * reward, next_state, next_action, done)
                if args.method == 'mpc_deter':
                    # print(step,next_action)
                    next_action += np.random.normal(0., 1.0*(0.999**(frame_idx+1)), size=(action_dim,))
                    # print(step,next_action)
                if len(model_replay_buffer) > batch_size:
                    model_optim.update_model(batch_size, mini_iter=config['model_iter'])
            elif base_method == 'hlt':
                next_action, rho = hybrid_policy(next_state)
                if args.method == 'hlt_deter':
                    next_action += np.random.normal(0., 1.0*(0.999**(frame_idx+1)), size=(action_dim,))
                model_replay_buffer.push(state, action, reward_scale * reward, next_state, next_action, done)
                replay_buffer.push(state, action, reward, next_state, mask)
                if len(replay_buffer) > batch_size:
                    # sac.update(batch_size)
                    sac.update(replay_buffer, batch_size)
                    model_optim.update_model(batch_size, mini_iter=config['model_iter'])

            state = next_state
            action = next_action
            episode_reward += reward
            frame_idx += 1

            if args.render:
                env.render(mode="human")


            if frame_idx % (max_frames//10) == 0:
                last_reward = rewards[-1][1] if len(rewards)>0 else 0
                print(
                    'frame : {}/{}, \t last rew: {}'.format(
                        frame_idx, max_frames, last_reward
                    )
                )
                if args.log:
                    print('saving model and reward log')
                    pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                    if base_method != 'mpc':
                        torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')
                    if base_method != 'sac':
                        torch.save(model.state_dict(), path + 'model_' + str(frame_idx) + '.pt')
#             print(episode_reward,done)
            if args.done_util:
                if done:
                    break
        if (len(replay_buffer) > batch_size) or (len(model_replay_buffer) > batch_size):
            print('ep rew', ep_num, episode_reward, frame_idx)
        rewards.append([frame_idx, episode_reward,ep_num])
        logger.add_scalar('train/returns', episode_reward, frame_idx)
        experiment.log_metric('train/returns', episode_reward, step=frame_idx)

        if ep_num % 10 == 0:
            print("EVALUATING")
            episode_rewards = []
            with torch.no_grad():
                for _ in range(10):
                    state = env.reset()
                    if base_method == 'sac':
                        action = policy_net.get_action(state)
                    elif base_method == 'mpc':
                        planner.reset()
                        action, _ = planner(state)
                    else:
                        hybrid_policy.reset()
                        action, _ = hybrid_policy(state)

                    episode_reward = 0
                    done = False
                    for step in range(max_steps):
                        for _ in range(frame_skip):
                            next_state, reward, done, _ = env.step(action.copy())

                        if base_method == 'sac':
                            next_action = policy_net.get_action(next_state)
                        elif base_method == 'mpc':
                            next_action, _ = planner(next_state)
                            if args.method == 'mpc_deter':
                                # print(step,next_action)
                                next_action += np.random.normal(0., 1.0*(0.999**(frame_idx+1)), size=(action_dim,))
                                # print(step,next_action)
                        elif base_method == 'hlt':
                            next_action, rho = hybrid_policy(next_state)
                            if args.method == 'hlt_deter':
                                next_action += np.random.normal(0., 1.0*(0.999**(frame_idx+1)), size=(action_dim,))

                        state = next_state
                        action = next_action
                        episode_reward += reward

                        if args.done_util:
                            if done:
                                break
                    episode_rewards.append(episode_reward)
            eval_rewards = np.mean(episode_rewards)
            print("EVAL REWARDS", eval_rewards)
            logger.add_scalar('eval/returns', eval_rewards, frame_idx)
            experiment.log_metric('eval/returns', eval_rewards, step=frame_idx)
        ep_num += 1
    env.close()
    if args.log:
        print('saving final data set')
        pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
        if base_method != 'mpc':
            torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
        if base_method != 'sac':
            torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')
