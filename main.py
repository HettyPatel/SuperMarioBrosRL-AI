
from mario_env.mario_env import create_mario_env
import sys 
sys.path.append("mario_env")
sys.path.append("memory")
sys.path.append("agents")
import os
from memory.replay_buffer import ReplayBuffer
import time
import torch
import pandas as pd 
import numpy as np 
import logging
from agents.DeepQAgent import DQNAgent   
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack

#append the path to the sys path
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    
def save_checkpoint(agent, level, episode, in_game_time_left):
    model_dir = 'checkpoints'
    level_dir = os.path.join(model_dir, level)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(level_dir):
        os.makedirs(level_dir)
        
    checkpoint_path = os.path.join(level_dir, f"checkpoint_{episode}_{in_game_time_left}.pth")
    torch.save(agent.q_network.state_dict(), checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")




if __name__ == '__main__':
    # levels to beat for speedrun
    
    metrics_df = pd.DataFrame(columns=['level', 'episode', 'total_reward', 'avg_loss', 'steps', 'in_game_time_left'])
    
    #train an agent on the following levels. 
    levels = ["SuperMarioBros-1-1-v0",
              "SuperMarioBros-1-2-v0",
              "SuperMarioBros-4-1-v0",
              "SuperMarioBros-4-2-v0",
              "SuperMarioBros-8-1-v0",
              "SuperMarioBros-8-2-v0",
              "SuperMarioBros-8-3-v0",
              "SuperMarioBros-8-4-v0"
    ]
    
    best_times = {level: 0 for level in levels}
    
    
    #create a csv 
    

    
    
    for level in levels:
        
        csv_file_path = f'H:\Code\Super Mario Bros AI\SuperMarioBrosRL-AI\csv_files'
        csv_file_path = os.path.join(csv_file_path, f"{level}.csv")
        


        
        if os.path.exists(csv_file_path):
            metrics_df = pd.read_csv(csv_file_path)
        else:
            metrics_df = pd.DataFrame(columns=['level', 'episode', 'total_reward', 'avg_loss', 'steps', 'in_game_time_left'])
        
        
        # create the environment
        env = create_mario_env(level, render_mode='human')
        env = ResizeObservation(env, shape=84)
        env = GrayScaleObservation(env)
        #env = FrameStack(env, num_stack=4)
        
        state_dim = np.prod(env.observation_space.shape) # number of states
        action_dim = env.action_space.n # number of actions
        
        
        #TODO Import settings from config file later 
        agent = DQNAgent(state_dim, action_dim, buffer_size=500000, batch_size=128, lr=0.0001, gamma=0.99, episilon=1.0, episilon_decay=0.999, min_episilon=0.01)
        
        num_episodes = 500
        
        for episode in range(num_episodes):
            state = env.reset()[0]
            state = np.reshape(state, [1, state_dim])
            #print("flattned shape", state.shape)
            
            done = False
            total_reward = 0
            total_loss = 0
            step_count = 0
            
            start_time = time.time()
            
            while not done:
                #env.render()
                action = agent.select_action(state)
                next_state, reward, done, trunc, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_dim])
                
                #print("next state shape", next_state.shape)
                #print("time left in the level", info['time'])
                
                # reward function edit
                
                # if info.get('flag_get', False): # if the flag is reached then give a reward of 1000
                #     reward += 1000
                # else:
                #     reward -= 1 # if the flag is not reached then give a penalty of -1 for speedrun \
                        
                # # if mario dies then give a penalty of -1000
                # if done:
                #     if info.get('flag_get', False) == False:
                #         reward -= 1000
                        
                    
                agent.store_experience(state, action, reward, next_state, done)
                loss = agent.train()
                if loss is not None:
                    total_loss += loss
                
                total_reward += reward
                state = next_state
                step_count += 1
            
            avg_loss = total_loss / step_count if step_count > 0 else 0
            
            in_game_time_left = info['time']
            
            
            metrics = {'level': level, 'episode': episode, 'total_reward': total_reward, 'avg_loss': avg_loss, 'steps': step_count, 'in_game_time_left': in_game_time_left}
            new_metrics_df = pd.DataFrame([metrics])
            metrics_df = pd.concat([metrics_df, new_metrics_df], ignore_index=True)
            
            logging.info(f"Level: {level}, Episode: {episode}, Total Reward: {total_reward}, Average Loss: {avg_loss:.4f}, Steps: {step_count}, In-Game Time Left: {in_game_time_left}")
            print(f"Episode: {episode}, Total Reward: {total_reward}, Average Loss: {avg_loss:.4f}, Steps: {step_count}, In-Game Time Left: {in_game_time_left}")
            
            #chaged this just in case it doesn't reach the flag in 500 episodes
            # if in_game_time_left > best_times[level] and info.get('flag_get', False) == True:
            #     best_times[level] = in_game_time_left
            #     save_checkpoint(agent, level, episode, in_game_time_left)
                
            if episode % 100 == 0:
                metrics_df.to_csv(csv_file_path, index=False)
                save_checkpoint(agent, level, episode, in_game_time_left) # save the checkpoint every 10 episodes
        env.close()
    metrics_df.to_csv(csv_file_path, index=False)

           
    
            
              
            
            
            
            
        
        