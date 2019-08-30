import numpy as np
import pandas as pd
from scipy.stats import chi2
import gym
from gym import spaces

class ALMEnv(gym.Env):
    """
    Custom Asset Liability Management environment, which follows gym interface
    Inputs are an asset value (scalar), a liability flow (numpy array of shape (T,))
    and a pandas DataFrame, with historical returns of available assets
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, T = 80, rate = 0.08):

        super(ALMEnv, self).__init__()

        self.asset = 10**6
        self.liability = chi2.pdf(np.linspace(0, 16, T + 1)[1:], 6)
        self.liab_PV = self.liability / (1 + rate) ** np.arange(1, T + 1)
        self.liability = self.liability * (self.asset / np.sum(self.liab_PV))

        # self.liability = pd.read_csv('liabilities.csv', sep = ';').iloc[:, 0].values
        self.historical_return = pd.read_csv('series.csv', sep = ';', decimal = ',', index_col = 0)

        self.present_asset = self.asset
        self.present_liability = self.liability

        self.action_space = spaces.Box(low = 0, high = 1, shape = (self.historical_return.shape[1],), dtype = np.float32)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = self.liability.shape, dtype = np.float32)

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
