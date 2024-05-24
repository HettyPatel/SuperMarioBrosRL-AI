from models.dqn import QNetwork
import torch 
import sys
sys.path.append("memory") # add the memory folder to the sys path so that we can import ReplayBuffer
from memory.replay_buffer import ReplayBuffer
import numpy as np



# Class for Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, lr, gamma, episilon, episilon_decay, min_episilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.episilon = episilon
        self.episilon_decay = episilon_decay
        self.min_episilon = min_episilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.memory = ReplayBuffer(buffer_size)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
    
    # picking an action
    # select random action at first, then select the best action depending on the episilon value
    
    def select_action(self, state):
        # if a random number between 0 and 1 is less than the episilon, then select a random action
        if np.random.rand() < self.episilon: 
            return np.random.choice(self.action_dim) # select a random action
        
        # if it is not then take the best action from the q_network based on the highest q value
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_values = self.q_network(state) # actions you can take in given state
            return np.argmax(action_values.cpu().data.numpy()) # select the best action based on the highest q value
        
    # store the experience in the replay buffer
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
    
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
        
        
        
        #
        
        q_values = self.q_network(states)
        q_values = q_values.squeeze(1)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        
        next_q_values = self.q_network(next_states).squeeze(1)    
        
        ##print("next_q_values before max:", next_q_values.shape)
        next_q_values = next_q_values.max(1)[0]
        #print("next_q_values after max:", next_q_values.shape)

        

        # print("rewards shape:", rewards.shape)
        # print("next_q_values shape:", next_q_values.shape)
        # print("dones shape:", dones.shape)


        
        
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = torch.nn.functional.mse_loss(q_values, expected_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # decay the episilon value or keep it at the minimum value
        self.episilon = max(self.episilon * self.episilon_decay, self.min_episilon)
        
        return loss.item()
        