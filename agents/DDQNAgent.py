from models.ddqn import ConvQNetwork
import torch 
import sys
sys.path.append("memory") # add the memory folder to the sys path so that we can import ReplayBuffer
from memory.replay_buffer import ReplayBuffer
import numpy as np


class DDQNAgent:
    def __init__(self, input_size, action_dim, buffer_size, batch_size, lr, gamma, epsilon, epsilon_decay, min_epsilon):
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = torch.device("cuda")
        
        self.memory = ReplayBuffer(buffer_size)
        self.q_network = ConvQNetwork(input_size, action_dim).to(self.device)
        self.target_network = ConvQNetwork(input_size, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
    
    
        
    # picking an action
    # select random action at first, then select the best action depending on the epsilon value
    
    def select_action(self, state):
        # if a random number between 0 and 1 is less than the epsilon, then select a random action
        if np.random.rand() < self.epsilon: 
            return np.random.choice(self.action_dim)
        
        # if it is not then take the best action from the q_network based on the highest q value
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_values = self.q_network(state) # actions you can take in given state
            return np.argmax(action_values.cpu().data.numpy()) # select the best action based on the highest q value
        
    # store the experience in the replay buffer
    def store_experience(self, state, action, reward, next_state, done):
        clipped_reward = np.clip(reward, -15, 15)
        self.memory.add(state, action, clipped_reward, next_state, done)
        
        
    def train(self):
            
            if len(self.memory) < self.batch_size:
                return
            
            # sample a batch of experiences from the replay buffer
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size) # collect a batch of experiences
            
            # convert to tensors and move to device
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # get the q values for the current state and next state
            q_values = self.q_network(states)
            #print("q values", q_values.shape)
            
            next_q_values = self.target_network(next_states)
            #print("next q values", next_q_values.shape)
            # get the q value for the action taken
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            next_q_values = next_q_values.max(1)[0]
            
            target_q_value = rewards + self.gamma * next_q_values * (1 - dones)
            
            # calculate the loss
            loss = torch.nn.MSELoss()(q_value, target_q_value)
            
            # optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            self.soft_update()
            
            return loss.item()
        
    def soft_update(self):
        tau = 0.001
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
            
    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        