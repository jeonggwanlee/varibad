import ad_envs
import mjrl
from mjrl.policies.gaussian_mlp import MLP
from wapper import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.batch_reinforce import BatchREINFORCE
from mjrl.algos.ppo_clip import PPO
from mjrl.algos.ppo_clip import PPO
from mjrl.utils.train_agent import train_agent
import os

import gym
import argparse
import time as timer

SEED = 500
#
e = GymEnv("half-cheetah-joint-v0")
policy = MLP(e.spec, hidden_sizes=(32, 32), seed=SEED)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
agent = PPO(e, policy, baseline, save_logs=True)



print("========================================")
print("Starting policy learning")
print("========================================")

ts = timer.time()
train_agent(job_name='beta_test',
            agent=agent,
            seed=SEED,
            niter=2000,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=5,
            sample_mode='trajectories',
            num_traj=10,
            save_freq=50,
            evaluation_rollouts=5)
print("time taken = %f" % (timer.time()-ts))