# environment for the SMB game 

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import gym
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT





def create_mario_env(level_name, render_mode):
    '''
    Create a mario environment with the given level name
    '''
    
    if render_mode == 'human':
        env = gym_super_mario_bros.make(level_name, render_mode='human', apply_api_compatibility=True)
    elif render_mode == 'rgb_array':
        env = gym_super_mario_bros.make(level_name, render_mode='rgb_array', apply_api_compatibility=True)
    else:
        env = gym_super_mario_bros.make(level_name, apply_api_compatibility=True)  
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    return env