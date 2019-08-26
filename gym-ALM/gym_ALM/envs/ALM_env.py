import numpy as np
import pandas as pd
import gym
from gym import spaces

class ALMEnv(gym.Env):
    """
    Custom Asset Liability Management environment, which follows gym interface
    Inputs are an asset value (scalar), a liability flow (numpy array of shape (T,))
    and a pandas DataFrame, with historical returns of available assets
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, asset, liability, historical_return):

        super(ALMEnv, self).__init__()

        self.asset = asset
        self.liability = liability
        self.historical_return = historical_return

        self.present_asset = asset
        self.present_liability = liability

        self.action_space = spaces.Box(low = 0, high = 1, shape = (historical_return.shape[1],), dtype = np.float32)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = liability.shape, dtype = np.float32)

    def step(self, action):
        sim_ret = np.random.multivariate_normal(mean = self.historical_return.mean(axis = 0), cov = pd.DataFrame.cov(self.historical_return))
        self.present_asset = self.present_asset * np.sum(sim_ret * action) - self.present_liability[0]
        self.present_liability = np.append(self.present_liability[1:], 0) * sim_ret[0]

        terminal = False
        if self.present_asset < 0 or np.sum(self.present_liability) == 0:
            terminal = True

        if self.present_asset >= 0:
            reward = 1
        else:
            reward = 0

        observation = self.present_liability / self.present_asset
        
        info = None

        return observation, reward, terminal, info

    def reset(self):
        self.present_asset = self.asset
        self.present_liability = self.liability
        return(self.present_liability / self.present_asset)

    def render(self, mode = 'human', close = False):
        pass
