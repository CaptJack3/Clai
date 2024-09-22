''' HERE IS ALL THE States Implementation
Including:
        State()
        State1D()
'''
import numpy as np
import enum
from data import Prices

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2

class State:
    def __init__(self, bars_count, commission_perc, reset_on_close, reward_on_close=True, volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        #TODO
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

        # More Definitions added :
        self.open_price= 0.0
        self.have_position = False



    def reset(self, prices, offset):
        assert isinstance(prices,Prices)
        assert offset >= self.bars_count - 1
        self._prices = prices
        self._offset = offset
        self.have_position = False
        self.open_price= 0.0 # Cause NO position yet


    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        if self.volumes:
            return (4 * self.bars_count + 1 + 1, )
        else:
            return (3*self.bars_count + 1 + 1, )

    def encode(self):
        """
        Convert current state into numpy array.
        """
        #TODO
        res = np.array() # Add shape ??
        index = 0
        offset = self.offset #60
        N = self.bars_count #5
        # The indexes should be [56,57,58,59,60]
        #first = offset-N+1        #56
        if self.volumes:
          for bar in range(self.bars_count):
            # if 5 bars -> 0,1,2,3,4
            data_index = offset-N+1+bar
            res[index] = self._prices.high[data_index]
            res[index+1] = self._prices.low[data_index]
            res[index+2] = self._prices.close[data_index]
            res[index+3] = self._prices.volume[data_index]
            res[index+4] = 1.0 * self.have_position
            res[index+5] = 0.0 # default, if not having open position
            if self.have_position:
              res[index+5] = (self._cur_close()/self.open_price) -1.0 # NOT SURE ABOUT IMPLEMENTATION
            index = index + 6
        else:
          for bar in range(self.bars_count):
            # if 5 bars -> 0,1,2,3,4
            data_index = offset-N+1+bar
            res[index] = self._prices.high[data_index]
            res[index+1] = self._prices.low[data_index]
            res[index+2] = self._prices.close[data_index]
            #res[index+3] = self._prices.volume[data_index]
            res[index+3] = 1.0 * self.have_position
            res[index+4] = 0.0 # default, if not having open position
            if self.have_position:
              res[index+4] = (self._cur_close()/self.open_price) -1.0
            index = index + 5


        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        # print(action)
        # print(Actions)
        # print(type(action), action)  # Check what action is before the assertion
        #assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close() # זה מחיר הסגירה של הנר האחרון והפתיחה של הנר הנוכחי
        if action == Actions.Buy and not self.have_position:
            #TODO - DONE
            self.have_position = True
            self.open_price = close
            reward = reward - self.commission_perc

        elif action == Actions.Close and self.have_position:
            #TODO
            self.have_position = False
            self.open_price = 0.0
            reward = reward - self.commission_perc
            done = done | self.reset_on_close # If we done the episode or define that each close we will reset
            if self.reward_on_close:
              reward += (close / self.open_price -1) * 100.0

        self._offset += 1
        last_close = close
        close = self._cur_close() # מחיר סגירה חדש
        done = done | (self._offset >= self._prices.close.shape[0] - 2 ) # I'm not sure what the correct number should be maybe -0 also work, but just for safety...
        # Also, No need to go to another position taking if we are at the end of the data, Can take bigger Safety factor ....

        if self.have_position:
          if not self.reward_on_close:
            # Need to add the reward
            inc = close/last_close -1 #TODO - Check if right
            reward += inc * 100.0

        return reward, done
''' ------------'''
''' STEP 4  '''
''' ------------'''
class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    def encode(self):
        ''' Maybe need to pass prices,offset ? ? '''
        # self.reset(prices,offset) - ''' This helped me somehere '''
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        if self.volumes:
            res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = (self._cur_close() - self.open_price) / self.open_price
        return res