import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transforms import TanhTransform, AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.optim import Adam
import random
import pickle

epsilon = 1e-6

def numpify(x):
	return x.detach().cpu().numpy()

class DeterministicPolicy(nn.Module):
    def __init__(self,state_dim,a_dim,h_dim,action_space=None):

        super().__init__()      
        self.net = nn.Sequential(nn.Linear(state_dim,h_dim),
                            nn.Tanh(),
                            nn.Linear(h_dim,h_dim),
                            nn.Tanh(),
                            nn.Linear(h_dim,a_dim),
                            nn.Tanh())
                
        self.a_dim = a_dim

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self,state):

        x = self.net(state)
        a = self.action_scale*x + self.action_bias
        
        return a
    
    def sample(self,state,eps=None):

        a = self.forward(state)
        log_prob = torch.zeros_like(a).sum(-1)

        return a,log_prob,a



    def reset_state(self,batch_size=None):
        # this is not a recurrent policy so we don't have any state to reset
        pass

    def np_policy(self,state):

        with torch.no_grad():
            state = torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float)
            action = self.forward(state)
        action = action.cpu().detach().numpy()

        return action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class TanhGaussianPolicy(nn.Module):

    def __init__(self,state_dim,a_dim,h_dim,action_space=None,device=torch.device('cuda:0'),std_scale=0.1):

        super().__init__()
        
        self.layers = nn.Sequential(nn.Linear(state_dim,h_dim),
                            nn.Tanh(),
                            nn.Linear(h_dim,h_dim),
                            nn.Tanh())
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,a_dim))
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,a_dim),nn.Softplus())
        self.std_scale = std_scale

        self.a_dim = a_dim
        self.device = device

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self,state):

        x = self.layers(state)
        mean = self.mean_layer(x)
        sig  = self.std_scale*self.sig_layer(x)

        return mean,sig

    def sample(self,state,eps=None):
        mean,std = self.forward(state)
        normal = Normal(mean,std)

        if eps is None:
            x_t = normal.rsample()
        else:
            x_t = mean + std*eps
        y_t = torch.tanh(x_t)
        action = y_t *self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def np_policy(self,state):
        with torch.no_grad():
            state = torch.tensor(state,device=self.device,dtype=torch.float32)
            action,_,_ = self.sample(state)
            action = numpify(action)
        return action

    def get_action(self, state):
        return self.np_policy(state)

    def reset_state(self,batch_size=None):
        # this is not a recurrent policy so we don't have any state to reset
        pass

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

class GaussianPolicy(nn.Module):

    def __init__(self,state_dim,a_dim,h_dim,action_space=None,device=torch.device('cuda:0'),std_scale=0.1):

        super().__init__()
        
        self.layers = nn.Sequential(nn.Linear(state_dim,h_dim),
                            nn.Tanh(),
                            nn.Linear(h_dim,h_dim),
                            nn.Tanh())
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,a_dim))
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,a_dim),nn.Softplus())
        self.std_scale = std_scale

        self.a_dim = a_dim
        self.device = device

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self,state):

        x = self.layers(state)
        mean = self.mean_layer(x)
        sig  = self.std_scale*self.sig_layer(x)

        return mean,sig

    def sample(self,state,eps=None):
        mean,std = self.forward(state)
        normal = Normal(mean,std)

        if eps is None:
            action = normal.rsample()
        else:
            action = mean + std*eps
        y_t = torch.tanh(x_t)
        action = y_t *self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t).sum(-1, keepdim=True)
        
        return action, log_prob, mean

    def np_policy(self,state):
        with torch.no_grad():
            state = torch.tensor(state,device=self.device,dtype=torch.float32)
            action,_,_ = self.sample(state)
            action = numpify(action)
        return action

    def reset_state(self,batch_size=None):
        # this is not a recurrent policy so we don't have any state to reset
        pass

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

class ValueNet(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNet, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, state_dim, a_dim, h_dim, layer_norm=False):
        super().__init__()
        if layer_norm:
            self.layers = nn.Sequential(nn.Linear(state_dim+a_dim,h_dim),nn.LayerNorm(h_dim),nn.ReLU(),
                                        nn.Linear(h_dim,h_dim),          nn.LayerNorm(h_dim),nn.ReLU(),
                                        nn.Linear(h_dim,1))

        else:
            self.layers = nn.Sequential(nn.Linear(state_dim+a_dim,h_dim),nn.ReLU(),
                                        nn.Linear(h_dim,h_dim),          nn.ReLU(),
                                        nn.Linear(h_dim,1))
    def forward(self, state, action):
        state_action = torch.cat([state,action],dim=-1)
        return self.layers(state_action)


class EnsembleQNetwork(nn.Module):
    '''
    This is going to be a REDQ-style critic (ensemble of Q functions)
    it needs to be able to call a random subset of critics
    it also needs to be able to call all critics
    '''
    def __init__(self, state_dim, a_dim, h_dim, n_members, layer_norm=False):
        super().__init__()
        self.Q_nets = nn.ModuleList([QNetwork(state_dim, a_dim, h_dim, layer_norm=layer_norm) for i in range(n_members)])
        self.n_members = n_members
        self.member_inds = np.arange(self.n_members)
        
    def forward(self,state,action,subset_size=None):

        if subset_size is None:
            inds = self.member_inds # everything
        else:
            inds = np.random.choice(self.member_inds,subset_size,replace=False) # random subset of size subset_size

        return torch.stack([self.Q_nets[i](state,action) for i in inds])

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def build_policy(state_dim,a_dim,args,action_space=None,device=torch.device('cuda:0')):
    if args.policy_type == 'deterministic':
        raise NotImplementedError
        policy = DeterministicPolicy(state_dim,a_dim,args.h_dim,action_space=action_space).to(device)
    elif args.policy_type == 'tanh_gaussian':
        policy = TanhGaussianPolicy(state_dim,a_dim,args.h_dim,action_space=action_space,std_scale=args.std_scale, device=device).to(device)
    elif args.policy_type == 'gaussian':
        policy = GaussianPolicy(state_dim,a_dim,args.h_dim,action_space=action_space,std_scale=args.std_scale).to(device)


    return policy

class SAC(object):
    def __init__(self,state_dim,action_space,args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.subset_size = args.q_subset_size # "M" parameter from REDQ https://arxiv.org/pdf/2101.05982.pdf
        self.args = args
        self.device = args.device
        self.updates = 0
        self.action_space = action_space

        self.policy_type = args.policy_type
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.policy = build_policy(state_dim,action_space.shape[0],args,action_space, device=args.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


        
                                            # state_dim, a_dim,                      h_dim,                        n_members,layer_norm=False
        self.critic        = EnsembleQNetwork(state_dim, action_space.shape[0], args.h_dim,n_members=args.q_ensemble_members,layer_norm=args.q_layer_norm).to(device=self.device)
        self.critic_target = EnsembleQNetwork(state_dim, action_space.shape[0], args.h_dim,n_members=args.q_ensemble_members,layer_norm=args.q_layer_norm).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        hard_update(self.critic_target, self.critic)

        if args.huber:
            self.QLoss = nn.HuberLoss()
        else:
            self.QLoss = nn.MSELoss()
        
        if self.args.automatic_entropy_tuning:

            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=4e-3)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size):
        self.update_parameters(replay_buffer, batch_size)

    def update_parameters(self, replay_buffer, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = replay_buffer.sample(batch_size=batch_size)

        state_batch      = torch.tensor(state_batch,     dtype=torch.float32,device=self.device)
        next_state_batch = torch.tensor(next_state_batch,dtype=torch.float32,device=self.device)
        action_batch     = torch.tensor(action_batch,    dtype=torch.float32,device=self.device)
        reward_batch     = torch.tensor(reward_batch,    dtype=torch.float32,device=self.device).unsqueeze(1)
        mask_batch       = torch.tensor(mask_batch,      dtype=torch.float32,device=self.device).unsqueeze(1)
        #### CRITIC LOSS ####
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf_next_target = self.critic_target(next_state_batch,next_state_action,subset_size=self.subset_size).min(0)[0] # take minimum across ensemble subset.  [0] selects values (as opposed to indecies).
            if not self.args.no_entropy_backup:
                # we are doing entropy backup
                qf_next_target -= self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * qf_next_target

        qfs = self.critic(state_batch,action_batch) # q values for each ensemble member, n_members x batch_size x state_dim x 1.
        qf_loss = self.QLoss(qfs,torch.stack(self.args.q_ensemble_members*[next_q_value]))
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        #### POLICY LOSS ####
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf_pi = self.critic(state_batch, pi).mean(0) # take mean across ensemble members

        policy_loss = ((self.alpha * log_pi) - qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        #### ALPHA LOSS ####
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), -torch.mean(log_pi).item()

    def save(self,path):
        torch.save({'policy_state_dict':self.policy.state_dict(),
                    'policy_optimizer_state_dict':self.policy_optim.state_dict(),
                    'critic_state_dict':self.critic.state_dict(),
                    'critic_target_state_dict':self.critic_target.state_dict(),
                    'critic_optimizer_state_dict':self.critic_optim.state_dict()
                    },path)

    def load(self,path):
        ckpt = torch.load(path)

        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.policy_optim.load_state_dict(ckpt['policy_optimizer_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.critic_target.load_state_dict(ckpt['critic_target_state_dict'])
        self.critic_optim.load_state_dict(ckpt['critic_optimizer_state_dict'])
        
class ReplayMemory:
    def __init__(self, capacity, seed):
        if seed is not None:
            random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, save_path):
        # if not os.path.exists('checkpoints/'):
        #   os.makedirs('checkpoints/')

        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        try:
            with open(save_path, "rb") as f:
                self.buffer = pickle.load(f)
                self.position = len(self.buffer) % self.capacity
        except:
            # for when we're loading from a .npz file
            data = np.load(save_path)
            states = data['states'] # N x T x s_dim
            actions = data['actions']
            next_states = data['next_states']
            rewards = data['rewards']

            N,T,_ = states.shape

            for i in range(N):
                for t in range(T):
                    s = states[i,t]
                    a = actions[i,t]
                    ns = next_states[i,t]
                    r = rewards[i,t].item()
                    if t == T-1:
                        m = 0
                    else:
                        m = 1
                    # ipdb.set_trace()

                    self.push(s,a,r,ns,m)




    