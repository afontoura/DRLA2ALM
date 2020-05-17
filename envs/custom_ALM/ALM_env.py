import numpy as np
import pandas as pd
import gym
from gym import spaces
from scipy.stats import chi2

"""
ALM Environment
This environment is not part of the original OpenAI SpinningUp package
It's been included by the author
"""

class ALM(gym.Env):
    """
    Custom Asset Liability Management environment, which follows gym interface
    Inputs are an asset value (scalar), a liability flow (numpy array of shape (T,))
    and a pandas DataFrame, with historical returns of available assets
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(ALM, self).__init__()

        self.T = 80
        self.rate = .08

        self.asset = 10**6
        self.liability = chi2.pdf(np.linspace(0, 16, 101)[(101 - self.T):], 6)
        self.liab_PV = self.liability / (1 + self.rate) ** np.arange(1, self.T + 1)
        self.liability = self.liability * (self.asset / np.sum(self.liab_PV))

        self.historical_return = pd.DataFrame(np.array([[1.277103375, 1.138939668, 1.196332479, 1.056897333],
                                                        [1.329337917, 1.220865211, 1.152575668, 1.031417750],
                                                        [1.436512041, 1.140436021, 1.119179339, 1.044573304],
                                                        [0.587765708, 1.110294883, 1.123874437, 1.059023134],
                                                        [1.826577896, 1.189505009, 1.099795940, 1.043120283],
                                                        [1.010439144, 1.170441620, 1.097598994, 1.059090683],
                                                        [0.818913771, 1.151082491, 1.116280628, 1.065031090],
                                                        [1.073968355, 1.266771188, 1.085152406, 1.058385690],
                                                        [0.845042000, 0.899819586, 1.081912809, 1.059108181],
                                                        [0.970877745, 1.145438240, 1.108091597, 1.064076166],
                                                        [0.866858640, 1.088815325, 1.132591303, 1.106734980],
                                                        [1.389351542, 1.248106035, 1.138430157, 1.062880551],
                                                        [1.268567254, 1.127940692, 1.101916922, 1.029473499],
                                                        [1.150323290, 1.130338666, 1.064204290, 1.037454821]]),
                                                        columns = ['Bovespa', 'IMA-B', 'IMA-S', 'IPCA'],
                                                        index = np.arange(2005, 2019))

        self.present_asset = self.asset
        self.present_liability = self.liability

        self.action_space = spaces.Box(low = 0, high = 1, shape = (self.historical_return.shape[1] - 1,), dtype = np.float32)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = self.liability.shape, dtype = np.float32)

    def step(self, action):
        # action = action + 1
        # action = np.exp(np.arctanh(action))
        # action = action / action.sum()
        sim_ret = np.random.multivariate_normal(mean = self.historical_return.mean(axis = 0), cov = pd.DataFrame.cov(self.historical_return))
        self.present_asset = self.present_asset * np.sum(sim_ret[:-1] * action) - self.present_liability[0]
        self.present_liability = np.append(self.present_liability[1:], 0) * sim_ret[-1]

        terminal = False
        if self.present_asset < 0 or np.sum(self.present_liability) == 0:
            terminal = True

        if self.present_asset >= 0:
            reward = 1
        else:
            reward = 0

        observation = self.present_liability / self.present_asset

        info = {'info': None}

        return observation, reward, terminal, info

    def reset(self):
        self.present_asset = self.asset
        self.present_liability = self.liability
        return(self.present_liability / self.present_asset)

    def render(self, mode = 'human', close = False):
        pass
