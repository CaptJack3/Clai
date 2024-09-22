import ptan
import numpy as np
import torch
import torch.optim as optim
from gym.utils import seeding
from tensorboardX import SummaryWriter
import importlib
import os
import data, common, validation
importlib.reload(common)
import torch
import pandas as pd
import enum
import gym
from States import State,State1D
from Trade_Env import StocksEnv
from Models import DQNConv1D
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
##

''' STEP 2 ISNT IMPORTANT TO THE FINAL RESULT'''

''' Step 2B: Define Constants and Enumerations'''
DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2
# For validation and understanding what is going on
print(len(Actions))
print(Actions.Close)
''' STEP 3 AND 4 ARE IMPLEMENTED IN States.py'''
''' STEP 5 IS IMPLEMENTED IN Trade_Env.py'''
''' Step 6A - Defining the cconvolutional model DQN is implemented in Models.py'''
''' STEP 6B - Parameters '''
#You can adjust the numbers depending on how long you want your training to be

BATCH_SIZE = 32
BARS_COUNT = 5 # 50
TARGET_NET_SYNC =100 # 1000
DEFAULT_STOCKS = "YNDX_160101_161231.csv"
DEFAULT_VAL_STOCKS = "YNDX_150101_151231.csv"
GAMMA = 0.99
REPLAY_SIZE = 1000  # 100000
REPLAY_INITIAL = 1000 # 10000
REWARD_STEPS = 2
LEARNING_RATE = 0.001 #0.0001
STATES_TO_EVALUATE = 100
EVAL_EVERY_STEP = 100
EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 1000 # 1000000
TOTAL_STEPS = 5000 #50000
CHECKPOINT_EVERY_STEP = 120 #24000
VALIDATION_EVERY_STEP = 600 #12000
args_data = DEFAULT_STOCKS
args_year = None  # Change to a specific year if needed
args_valdata = DEFAULT_VAL_STOCKS
''' STEP 6C - Saving Routine, addapted to local usage with ChatGpt '''
import os
from datetime import datetime
# Get the current date and time in the specified format
current_datetime = datetime.now().strftime("%Y%m%d-%H%M")
# Create the run name
args_run = f"test_run_{current_datetime}"
print(f"Run name is: {args_run}")
# Define the local save path
saves_path = os.path.join("model", args_run)
# Create the directory
os.makedirs(saves_path, exist_ok=True)
print(f"Saving to: {saves_path}")
''' Step 6D - Set up the environment and initialize the DQN agent for training on stock data : '''

''' SetUP arguments '''
N1 = 100 # Was 1000

if args_year is not None or os.path.isfile(args_data):
    if args_year is not None:
        stock_data = data.load_year_data(args_year)
    else:
        stock_data = {"YNDX": data.load_relative(args_data)}
    env = StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=True, volumes=False)
    env_tst = StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=True)
elif os.path.isdir(args_data):
    env = StocksEnv.from_dir(args_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=True)
    env_tst = StocksEnv.from_dir(args_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=True)
else:
    raise RuntimeError("No data to train on")
env = gym.wrappers.TimeLimit(env, max_episode_steps=N1)

val_data = {"YNDX": data.load_relative(args_valdata)}
env_val = StocksEnv(val_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=True)

writer = SummaryWriter(comment="-conv-" + args_run)
net = DQNConv1D(env.observation_space.shape, env.action_space.n).to(device)
print(net)
tgt_net = ptan.agent.TargetNet(net)
selector = ptan.actions.EpsilonGreedyActionSelector(EPSILON_START)
agent = ptan.agent.DQNAgent(net, selector, device=device)
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
# # Jacob:
# exp_source.pool[0].reset()
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

step_idx = 0
eval_states = None
best_mean_val = None

with common.RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
    #while True:
    while step_idx < TOTAL_STEPS:
        step_idx += 1
        if step_idx%200==0:
            print(f" @@@@@ The step_idx is {step_idx}")
        buffer.populate(1)
        selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)

        new_rewards = exp_source.pop_rewards_steps()
        if new_rewards:
            reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)

        if len(buffer) < REPLAY_INITIAL:
            continue

        if eval_states is None:
            print("Initial buffer populated, start training")
            eval_states = buffer.sample(STATES_TO_EVALUATE)
            eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
            eval_states = np.array(eval_states, copy=False)

        if step_idx % EVAL_EVERY_STEP == 0:
            mean_val = common.calc_values_of_states(eval_states, net, device=device)
            writer.add_scalar("values_mean", mean_val, step_idx)
            if best_mean_val is None or best_mean_val < mean_val:
                if best_mean_val is not None:
                    print("%d: Best mean value updated %.3f -> %.3f" % (step_idx, best_mean_val, mean_val))
                best_mean_val = mean_val
                torch.save(net.state_dict(), os.path.join(saves_path, "mean_val-%.3f.data" % mean_val))

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_v = common.calc_loss(batch, net, tgt_net.target_model, GAMMA ** REWARD_STEPS, device=device)
        loss_v.backward()
        optimizer.step()

        if step_idx % TARGET_NET_SYNC == 0:
            tgt_net.sync()

        if step_idx % CHECKPOINT_EVERY_STEP == 0:
            idx = step_idx // CHECKPOINT_EVERY_STEP
            torch.save(net.state_dict(), os.path.join(saves_path, "checkpoint-%3d.data" % idx))

        if step_idx % VALIDATION_EVERY_STEP == 0:
            print(f"$$$$$$$$$ entered Validation_EVERY_STEP -----")
            res = validation.validation_run(env_tst, net, Actions, device=device)
            for key, val in res.items():
                writer.add_scalar(key + "_test", val, step_idx)
            res = validation.validation_run(env_val, net, Actions, device=device)
            for key, val in res.items():
                writer.add_scalar(key + "_val", val, step_idx)
    print("I'm here # 12345")

print("***************************************************")
print("-----------        FINISH  ------------------------")
print("***************************************************")


