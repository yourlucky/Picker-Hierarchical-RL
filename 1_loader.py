#작동하는 놈 
#source /Users/yoon/Documents/python/FitML/RL_stable/bin/activate

import gym
import pybullet
import pybullet_envs
import torch as th

import stable_baselines3 as sb3
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy

#env = gym.make("HumanoidBulletEnv-v0")
env = gym.make("AntBulletEnv-v0")

#redering or not
env.render()
model = sb3.PPO.load("1.sub_policy/TRPO/mlp/trpo_Mlp+100",env)
#model = TRPO.load("TRPO/mlp/trpo_Mlp+14",env)
#model = sb3.SAC.load("SAC/mlp/sac_Mlp+28",env)

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
#print("=============================================================")
#print("@ mean_reward : " , mean_reward)
#print("@ std_reward : " , std_reward)


# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# Watch Video
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    if dones:
        print(dones)
    env.render()
env.close()
