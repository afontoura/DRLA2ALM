# Main packages
import gym
import envs
import numpy as np

# DDPG classes
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

# TD3 classes
from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# Variables setup
seeds = [10, 20, 30, 40, 50]
rates = [.06, .065, .07, .075, .08]
time_horizons = [30, 50, 80]

# Main loop
for seed in seeds:
    for discount_rate in rates:
        for time_horizon in time_horizons:

            # Common variables
            env = gym.make('ALM-v0', T = time_horizon, rate = discount_rate)
            n_actions = env.action_space.shape[-1]

            # DDPG
            param_noise = None
            action_noise_ddpg = OrnsteinUhlenbeckActionNoise(mean = np.zeros(n_actions), sigma = float(0.5) * np.ones(n_actions))

            model = DDPG(MlpPolicy, env, verbose = 0, param_noise = param_noise, action_noise = action_noise_ddpg, seed = seed)
            model.learn(total_timesteps = 500000)

            model.save('ddpg_t' + str(time_horizon) + '_r' + str(round(discount_rate * 1000)) + '_seed' + str(seed))

            # TD3
            action_noise_td3 = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions))

            model = TD3(MlpPolicy, env, action_noise = action_noise_td3, verbose = 0, seed = seed)
            model.learn(total_timesteps = 500000, log_interval = 100)

            model.save('td3_t' + str(time_horizon) + '_r' + str(round(discount_rate * 1000)) + '_seed' + str(seed))
