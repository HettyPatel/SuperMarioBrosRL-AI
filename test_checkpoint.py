import os
import gym
import torch
import time
import numpy as np
from mario_env.mario_env import create_mario_env
from agents.DDQNAgent import DDQNAgent
from gym.wrappers import ResizeObservation, GrayScaleObservation

# Function to load the checkpoint
def load_checkpoint(agent, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        agent.target_network.load_state_dict(checkpoint['model_target_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #agent.epsilon = checkpoint['epsilon']
        agent.epsilon = 0.00  # Set epsilon to 0.00 to start from the checkpoint
        print("Checkpoint loaded successfully from", checkpoint_path)
    else:
        print("Checkpoint not found at", checkpoint_path)

class SkipFrames(gym.Wrapper):
    def __init__(self, env, skip):
        
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        
        for _ in range(self._skip):
            state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break
        return state, total_reward, done, trunc, info

# Create the environment
level = "SuperMarioBros-1-1-v0"
env = create_mario_env(level, render_mode='human')
env = SkipFrames(env, skip=4)
env = ResizeObservation(env, shape=84)
env = GrayScaleObservation(env)

# Initialize the agent
state_dim = (1, 84, 84)  # Assuming grayscale and resized observation
action_dim = env.action_space.n
agent = DDQNAgent(state_dim, action_dim, buffer_size=1000, batch_size=32, lr=0.00025, gamma=0.99, epsilon=1, epsilon_decay=0.999, min_epsilon=0.1)

# Load the agent from checkpoint
checkpoint_path = 'H:\Code\Super Mario Bros AI\SuperMarioBrosRL-AI\checkpoints\DDQN FINISHD\DDQN\SuperMarioBros-1-1-v0\latest_checkpoint.pth'
load_checkpoint(agent, checkpoint_path)

# Run the agent
state = env.reset()
done = False
total_loss = 0
total_reward = 0
step_count = 0


num_episodes = 100
success_count = 0
framedelay = 0.009

for episode in range(num_episodes):
    state = env.reset()[0]
    state = np.expand_dims(state, axis=0)
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, trunc, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        agent.store_experience(state, action, reward, next_state, done)
        
        if done:
            if info.get('flag_get', False):
                success_count += 1
                #time = info['time']
                #print("time to complete the level:", 400-time)
                #print("Success count:", success_count)        
        
            
        state = next_state
        total_reward += reward
        step_count += 1
        time.sleep(framedelay)
        if done:
            break
    
env.close()

print("Total Success count:", success_count)
print("Total reward:", total_reward)
print("Total steps:", step_count)

