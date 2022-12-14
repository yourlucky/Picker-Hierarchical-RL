#작동하는 놈 -Ant만
#source /Users/yoon/Documents/python/FitML/RL_stable/bin/activate
import pandas as pd
import numpy as np

import gym
import pybullet
import pybullet_envs
import torch as th

import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy


env = gym.make("AntBulletEnv-v0")
#env = gym.make("HumanoidBulletEnv-v0")
#env.render()

MAX_AVERGAE_SCORE = 2710


#model = sb3.SAC('MlpPolicy',env)
model = sb3.SAC.load("sac_Mlp+50",env)
#model = sb3.SAC('CnnPolicy',env)
#MultiInputPolicy
_index=[]
_mean=[]
_std=[]

for i in range(51,101):
    print("Training itteration ",i)
    model.learn(total_timesteps=10000)
    if i<51 and i%2==0:
        k="sac_Mlp+"+str(i)
        model.save(k)
    model.save("z_sac_Mlp_100")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    _index.append(i)
    _mean.append(mean_reward)
    _std.append(std_reward)
    #if(i%10 == 0):
        #print("mean_reward : ", mean_reward)
    #if mean_reward >= MAX_AVERGAE_SCORE:
        #break
del model

raw_data = {'index': _index,
            'mean': _mean,
            'std': _std}
df = pd.DataFrame(raw_data, columns = ['index','mean', 'std'])
df.to_csv('sac_Mlp_50_100.csv', index=False, header=True)