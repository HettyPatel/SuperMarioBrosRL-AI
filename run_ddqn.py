from mario_env.mario_env import create_mario_env
import sys 
sys.path.append("mario_env")
sys.path.append("memory")
sys.path.append("agents")
from memory.replay_buffer import ReplayBuffer
import time
import torch
import pandas as pd 
import numpy as np 
import gym
import os
import logging
from agents.DDQNAgent import DDQNAgent   
from gym.wrappers import ResizeObservation, GrayScaleObservation

#append the path to the sys path
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    
import shutil

def save_checkpoint(agent, level, episode, in_game_time_left):
    model_dir = 'checkpoints/DDQN'
    level_dir = os.path.join(model_dir, level)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(level_dir):
        os.makedirs(level_dir)

    checkpoint_path = os.path.join(level_dir, f"checkpoint_{episode}_{in_game_time_left}.pth")
    checkpoint_data = {
        'episode': episode,
        'model_state_dict': agent.q_network.state_dict(),
        'model_target_state_dict': agent.target_network.state_dict(),  # corrected typo here
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'in_game_time_left': in_game_time_left
    }
    torch.save(checkpoint_data, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")

    # Update the 'latest_checkpoint.pth' by copying the checkpoint file
    latest_checkpoint_path = os.path.join(level_dir, 'latest_checkpoint.pth')
    shutil.copy(checkpoint_path, latest_checkpoint_path)
    
def load_checkpoint(agent, level, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        agent.q_network.load_state_dict(checkpoint['model_state_dict'])
        agent.target_network.load_state_dict(checkpoint['model_target_state_dict'])  # corrected typo here
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']  # corrected typo here
        return checkpoint['episode'], checkpoint['in_game_time_left']
    else:
        return 0, 0  # If no checkpoint exists, start from scratch


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
    

metrics_df = pd.DataFrame(columns=['level', 'episode', 'total_reward', 'avg_loss', 'steps', 'in_game_time_left'])

levels = ["SuperMarioBros-1-1-v0",
          "SuperMarioBros-1-2-v0",
           "SuperMarioBros-4-1-v0",
            "SuperMarioBros-4-2-v0",
            "SuperMarioBros-8-1-v0",
            "SuperMarioBros-8-2-v0",
            "SuperMarioBros-8-3-v0",
           "SuperMarioBros-8-4-v0"
            ]

for level in levels:
    csv_file_path = f'H:\Code\Super Mario Bros AI\SuperMarioBrosRL-AI\csv_files\DDQN'
    csv_file_path = os.path.join(csv_file_path, f"{level}.csv")
    if os.path.exists(csv_file_path):
        metrics_df = pd.read_csv(csv_file_path)
    else:
        metrics_df = pd.DataFrame(columns=['level', 'episode', 'total_reward', 'avg_loss', 'steps', 'in_game_time_left'])
        
    
    env = create_mario_env(level, render_mode='human')
    env = SkipFrames(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    #env = gym.wrappers.FrameStack(env, num_stack=4)
    
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    in_size = (1, 84, 84)
  
    
    agent = DDQNAgent(in_size, action_dim, buffer_size=500000, batch_size=64, lr=0.00025, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.1)
    num_episodes = 1000
    
    checkpoint_path = os.path.join('checkpoints', 'DDQN', level, 'latest_checkpoint.pth')
    start_episode, in_game_time_left = load_checkpoint(agent, level, checkpoint_path)
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        
        #print("state shape", state.shape)
        
        done = False
        total_reward = 0
        total_loss = 0
        step_count = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, trunc, info = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            #print("next state shape", next_state.shape)
            
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                total_loss += loss
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                break
            
        avg_loss = total_loss / step_count if step_count > 0 else 0
            
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
        logging.info(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {step_count}, Epsilon: {agent.epsilon}")
        print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {step_count}, Epsilon: {agent.epsilon}")
        metrics = {'level': level, 'episode': episode, 'total_reward': total_reward, 'avg_loss': avg_loss, 'steps': step_count, 'in_game_time_left': in_game_time_left}
        new_metrics_df = pd.DataFrame([metrics])
        metrics_df = pd.concat([metrics_df, new_metrics_df], ignore_index=True)
        
        if episode % 100 == 0:
            metrics_df.to_csv(csv_file_path, index=False)
            save_checkpoint(agent, level, episode, in_game_time_left)
            
            # Update the latest checkpoint to the latest saved checkpoint
            checkpoint_dir = os.path.join('checkpoints', 'DDQN', level)
            latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            checkpoint_filename = f'checkpoint_{episode}_{in_game_time_left}.pth'
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            
            if os.path.exists(latest_checkpoint_path):
                os.remove(latest_checkpoint_path)
            
            # Copy the checkpoint file to latest_checkpoint.pth
            shutil.copy(checkpoint_path, latest_checkpoint_path)
           
           
        
        