import gym #0.26, 0.21<-stable
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pybullet
import pybullet_envs
import stable_baselines3 as sb3
from sb3_contrib import TRPO

#from torch.utils.tensorboard import SummaryWriter
import datetime
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env = gym.make('AntBulletEnv-v0')
#env = gym.make('CartPole-v1')
# Get number of actions from gym action space
#n_actions = env.action_space
n_actions = 3
#n_actions = 4
#state, _ = env.reset(return_info=True)
state = env.reset()

# Get the number of state observations
#n_observations = env.observation_space.shape[0]
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state,test=False):
    global steps_done
    if test:
        k=policy_net(state).argmax()
        return int(k)

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            k=policy_net(state).argmax()
            return int(k)
    else:
        k=np.random.randint(3)
        return int(k)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
 
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    if state_batch.ndim >=3:
        state_batch = torch.squeeze(state_batch)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    next_state_values = next_state_values.unsqueeze(1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



# if torch.cuda.is_available():
#     num_episodes = 600
# else:
#     num_episodes = 50
##evenly test_set
#model_0 = sb3.PPO.load("PPO/ac/ppo_ac+38",env)#996
#model_1 = TRPO.load("TRPO/mlp/trpo_Mlp+16",env)#963
#model_2 = sb3.SAC.load("SAC/ac/sac_ac+50",env)#983

##skewed test_set
model_0 = sb3.PPO.load("1.sub_policy/PPO/mlp/ppo_mlp+30",env)#2024
model_1 = TRPO.load("1.sub_policy/TRPO/ac/trpo_Mlp_vp+40",env)#1000
model_2 = sb3.SAC.load("1.sub_policy/SAC/ac/sac_ac+20",env)#557



#writer = SummaryWriter('./runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

_index=[]
_mean=[]
_std=[]
_count={'ppo':0,'trpo':0,'sac':0}
picking_term = 6
for i_episode in range(1,101):
    print(i_episode)
    if i_episode%2==0:
        static_reward=[]
        for k in range(5):
            state = env.reset()
            total_reward = 0
            obs = torch.tensor(state, dtype=torch.float32, device=device)
            for t in count():
                pick_action = select_action(obs,True)
                #pick_action = select_action(obs,True) #test면
                privious_state=obs
                picker_reward = 0

                for ii in range(picking_term):
                    if state.ndim>1:
                        state = state[0]
                    if pick_action==0:
                        action = model_0.predict(state, deterministic=True)
                        _count['ppo']+=1
                    elif pick_action==1:
                        action = model_1.predict(state, deterministic=True)
                        _count['trpo']+=1
                    elif pick_action==2:
                        action = model_2.predict(state, deterministic=True)
                        _count['sac']+=1

                    action=np.array(action[0],dtype=np.float32)

                    right_before_state = state
                    state, rewards, dones, info = env.step(action)
                    picker_reward +=rewards
                    total_reward+=rewards
                    if dones:
                        break
                if dones:
                    picker_next_state = None
                else:
                    picker_next_state = state        
                    picker_next_state=torch.tensor(picker_next_state, dtype=torch.float32, device=device)
                state = picker_next_state
                picker_reward=torch.tensor(picker_reward, dtype=torch.float32, device=device)
                pick_action=torch.tensor(pick_action, dtype=torch.int64, device=device)
                privious_state=torch.tensor(privious_state, dtype=torch.float32, device=device)
                if pick_action.ndim==0:
                    pick_action=torch.unsqueeze(pick_action, 0)
                if picker_reward.ndim==0:
                    picker_reward=torch.unsqueeze(picker_reward, 0)
            

                if dones:
                    break
            static_reward.append(total_reward)
        mean_reward=np.mean(static_reward)
        std_reward=np.std(static_reward)
        
        print("reward : mean",mean_reward)
        print("reward : std",std_reward)
        print("raw : ",static_reward)
        print("count : ",_count)
        _index.append(i_episode)
        _mean.append(mean_reward)
        _std.append(std_reward)

    state = env.reset()
    total_reward = 0
    obs = torch.tensor(state, dtype=torch.float32, device=device)
    for t in count():
        pick_action = select_action(obs)
        #pick_action = select_action(obs,True) #test only
        privious_state=obs
        picker_reward = 0

        for ii in range(picking_term):
            if state.ndim>1:
                state = state[0]
            if pick_action==0:
                action = model_0.predict(state, deterministic=True)
            elif pick_action==1:
                action = model_1.predict(state, deterministic=True)
            elif pick_action==2:
                action = model_2.predict(state, deterministic=True)

            action=np.array(action[0],dtype=np.float32)

            right_before_state = state
            state, rewards, dones, info = env.step(action)
            picker_reward +=rewards
            total_reward+=rewards
            if dones:
                break
        if dones:
            picker_next_state = None
        else:
            picker_next_state = state        
            picker_next_state=torch.tensor(picker_next_state, dtype=torch.float32, device=device)
        state = picker_next_state
        picker_reward=torch.tensor(picker_reward, dtype=torch.float32, device=device)
        pick_action=torch.tensor(pick_action, dtype=torch.int64, device=device)
        privious_state=torch.tensor(privious_state, dtype=torch.float32, device=device)
        if pick_action.ndim==0:
            pick_action=torch.unsqueeze(pick_action, 0)
        if picker_reward.ndim==0:
            picker_reward=torch.unsqueeze(picker_reward, 0)

        
        #memory.push(privious_state, pick_action, picker_next_state, picker_reward)
        if picker_next_state is not None:
            memory.push(torch.unsqueeze(privious_state, 0),torch.unsqueeze(pick_action, 0),torch.unsqueeze(picker_next_state, 0),torch.unsqueeze(picker_reward, 0))
        else:
            memory.push(torch.unsqueeze(privious_state, 0),torch.unsqueeze(pick_action, 0),None,torch.unsqueeze(picker_reward, 0))
        

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if dones:
            #episode_durations.append(t + 1)
            break




#save the data in csv file
'''
raw_data = {'index': _index,
            'mean': _mean,
            'std': _std}

df = pd.DataFrame(raw_data, columns = ['index','mean', 'std'])
df.to_csv('skew_6.csv', index=False, header=True)

with open('skew_Selection_6.csv','w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=_count)
    writer.writeheader()
    writer.writerow(_count)
'''
print('Complete')
