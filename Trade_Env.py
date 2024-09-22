''' -----------------------------'''
'''     STEP 5 >>'''
''' -----------------------------'''
import enum
import random
import gym
import numpy as np
from States import State,State1D
import ptan
import torch
import torch.optim as optim
from gym.utils import seeding
from tensorboardX import SummaryWriter
import importlib
import os


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1

class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    # spec = ... # Don't we need it ?

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC, reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False, volumes=False):
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                  volumes=volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=3)  # Can be n=len(Actions)
        self.observation_space = gym.spaces.Box(low = - np.inf , high = np.inf , shape=self._state.shape , dtype=np.float32) #TODO (check gym.spaces.Box)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        #TODO
        # Jacob : We can have few instruments. the instruments are the keys of the prices dictionary
        ''' Selection of Instrument '''
        # I suppose that the selection is random
        self.instrument = random.choice(list(self._prices.keys())) # NOT SURE IF BEST IMPLEMENTATION, NO SEED ?
        ''' define the prices'''
        prices = self._prices[self.instrument]
        bars_count = self._state.bars_count
        if not self.random_ofs_on_reset:
          # From befining
          offset = bars_count -1 #(Need to check this)
        else:
          offset = self.np_random.choice(prices.high.shape[0]-bars_count*10) + bars_count #TODO # UNDERSTAND
        # print(f"$$$ offset is  {offset}")

        self._state.reset(prices,offset)

        return self._state.encode()

    def step(self, action_idx):
        #TODO - DONE
        action = Actions(action_idx)
        reward , done = self._state.step(action)
        obs = self._state.encode()
        ''' Adding all posible data I have '''
        info = dict()
        #info["state1d"] = state_1d #TODO = I had some problem here, don't sure if needed...
        info["random_ofs_on_reset"] = self.random_ofs_on_reset
        info["instrument"] = self.instrument
        info["offset"] = self._state._offset #TODO - Check ??


        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)

