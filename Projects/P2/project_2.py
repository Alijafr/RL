#!/usr/bin/env python
# coding: utf-8



from collections import deque, namedtuple
import random
import numpy as np
import gym 
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt 
from tqdm import tqdm


class Deep_Q_Network(nn.Module):
    def __init__(self,num_states,num_actions,nodes_1 =50, nodes_2 = 50,seed =10):
        super(Deep_Q_Network,self).__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(num_states,nodes_1)
        self.fc2 = nn.Linear(nodes_1,nodes_2)
        self.fc3 = nn.Linear(nodes_2,num_actions)
    
    def forward(self,states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class MemoryReplay:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size,seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    
    def append(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self,batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None]).squeeze(1)
        rewards = np.vstack([e.reward for e in experiences if e is not None]).squeeze(1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).squeeze(1)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



class DQL_Agent:
    def __init__(self,env,memory_max_size =10_000,dicount= 0.99,lr_optim=1e-3,DQL_node1=50,DQL_node2=50,decay_rate = 0.996,seed =10):
        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_action = env.action_space.n
        self.dicount = dicount
        self.seed = seed
        self.eps = 1.0
        self.decay_rate_eps = decay_rate
        self.min_eps = 0.05
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env.seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.reply_memory = MemoryReplay(memory_max_size,self.seed)
        self.Q_action = Deep_Q_Network(self.num_states,self.num_action,DQL_node1,DQL_node2,self.seed).to(self.device)
        self.Q_target = Deep_Q_Network(self.num_states,self.num_action,DQL_node1,DQL_node2,self.seed).to(self.device)
        self.Q_target.eval() #will turn off any dropout or batch norm layer 
        #make sure both network has identical weights 
        self.update_target_weights()
        
        self.loss_fucntion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.Q_action.parameters(),lr=lr_optim)
        
        
    def update_target_weights(self):
        self.Q_target.load_state_dict(self.Q_action.state_dict())
    
    def eps_greedy(self,states):
        if np.random.rand()<self.eps:
            return self.env.action_space.sample()
        else:
            #act greedy
            
            #make sure the state are tensor in order to feed it to the network
            if not torch.is_tensor(states):
                states = torch.from_numpy(states[np.newaxis,:]).float().to(self.device)
            with torch.no_grad(): #will disable tracking the gradient --> reduce cpu/memory usage
                action = self.Q_action(states)
            max_action = torch.argmax(action).item()
            return max_action
    
    def decay_eps(self):
        self.eps = np.maximum(self.eps*self.decay_rate_eps,self.min_eps)
    
    def to_tensor(self,states, actions, rewards,next_states,Dones):
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        Dones = torch.from_numpy(Dones).to(self.device)
        return states, actions, rewards, next_states, Dones
    def learnFromExperience(self,miniBatchSize): #hallucinations
        if miniBatchSize <2:
            raise ValueError("batch size must greater than 1")
        #make sure we have enough experiences 
        if len(self.reply_memory) < miniBatchSize:
            return #not enough experience, sounds familiar right?
        #else sample and learn
        states, actions, rewards, next_states, Dones = self.reply_memory.sample(miniBatchSize)
        #convert the result to tensor for model input 
        states, actions, rewards, next_states, Dones = self.to_tensor(states, actions, rewards, next_states, Dones)
        #calculate the current Q estimation 
        Q_estimate = self.Q_action(states)
        #obtain the q value for the actioned used in the experiences 
        Q_estimate_a = Q_estimate.gather(1, actions.view(-1, 1)).squeeze(1)
        
        #calculate the target value using --> rewards + discount* argmax_a Q(next_state, target_network_weight)
        #the max gives both the max values and the indices 
        Q_target = self.Q_target(next_states).max(dim=1).values
        #note that one the state is terminal, we only count the reward, therefore, we need to check if the state is Dones
        #if Done is true, we should not calculate Q for the next states 
        Q_target[Dones] = 0.0 
        #final target calculation
        Q_target = rewards + self.dicount*Q_target
        
        #make sure the grad is zero 
        self.optimizer.zero_grad()

        #calculate the loss 
        loss=self.loss_fucntion(Q_target,Q_estimate_a)
        #calcualte the gradient dL/dw
        loss.backward()
        #optimize using gradient decent
        self.optimizer.step()
        
    def get_max_action(self,states):
        self.Q_action.eval()
        #make sure the state are tensor in order to feed it to the network
        if not torch.is_tensor(states):
            states = torch.from_numpy(states[np.newaxis,:]).to(self.device)
        with torch.no_grad(): #will disable tracking the gradient --> reduce cpu/memory usage
            action = self.Q_action(states)
        max_action = torch.argmax(action).item()
        return max_action
        
    def save_model(self,path):
        torch.save(self.Q_action.state_dict(), path) 
    
    def load_model(self,path):
        self.Q_action.load_state_dict(torch.load(path))
            


class Training_agent:
    #Agent and environment interaction
    def __init__(self, env,memory_max_size =10_000,discount= 0.99,lr_optim=0.00065,update_freq =1000,DQL_node1=88,DQL_node2=50,decay_rate=0.991):
        self.seed =77502
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        env.seed(self.seed)
        self.agent = DQL_Agent(env,memory_max_size ,discount ,lr_optim,DQL_node1,DQL_node2,decay_rate,self.seed)
        self.update_freq = update_freq
        self.steps = 0 
    
    def train_agent(self, num_episodes,batch_size):
        self.rewards = np.zeros(num_episodes)
        self.moving_average = []
        for i in tqdm(range(num_episodes)):
            state = self.agent.env.reset()
            Done = False
            total_rewards = 0
            n = 0
            while not Done: 
                n +=1
                #take an action using the greedy policy
                action = self.agent.eps_greedy(state)
                #implement the action 
                next_state, reward, Done, info = self.agent.env.step(action)
                #save the experience in the memory of the agent 
                self.agent.reply_memory.append(state,action,reward,next_state,Done)
                #sum the rewards
                total_rewards += reward
                
                #learn from experience (if there is enough)
                self.agent.learnFromExperience(batch_size)
                #update the tarqet network per the desired frequency 
                self.steps +=1 
                if (self.steps % self.update_freq) == 0:
                    self.agent.update_target_weights()
                state = next_state
            #append the rewards
            self.rewards[i] = total_rewards
            self.moving_average.append(np.mean(self.rewards[-50:]))
            #update the eps 
            self.agent.decay_eps()
            if i %10 == 0:
                print("The episode {} total rewards is {}".format(i+1, total_rewards))
                print(len(self.agent.reply_memory))
    def test_agent(self,num_run, render=False):
        rewards = np.zeros(num_run)
        for i in tqdm(range(num_run)):
            state = self.agent.env.reset()
            Done = False 
            total_rewards = 0

            while not Done: 
                action = self.agent.get_max_action(state)
                next_state, reward,Done, info = self.agent.env.step(action)
                total_rewards += reward
                if render:
                    self.agent.env.render()
                state = next_state
            rewards[i]= total_rewards
            #print("The episode total rewards is ", total_rewards)
        return rewards
        


num_eps = 700


#Main training 

import time
env = gym.make('LunarLander-v2')
training_agent = Training_agent(env)
start = time.time()
training_agent.train_agent(num_eps,64)
print ((time.time()-start)/60.)


#whole model 
model_scripted = torch.jit.script(training_agent.agent.Q_action) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save


# In[ ]:


#training_agent.agent.Q_action.load_state_dict(torch.load("model.pt"))

#whole model 
#model = torch.jit.load('model_scripted.pt')
#model.eval()


# # Experiments 

# ## Experiemnt 1: traininig rewards + 50 moving average 

# In[43]:


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
moving_average_period = 50
moving_avg = moving_average(training_agent.rewards,n=moving_average_period)


# In[90]:


x = np.arange(moving_average_period,num_eps+1)
fig = plt.figure(figsize=(6, 4))
plt.plot(training_agent.rewards)
plt.plot(x,moving_avg,label='50 episodeing moving average')
plt.xlabel('number of episodes')
plt.ylabel('rewards')
plt.title('Training DQL agent')
plt.legend()
plt.savefig('training_500_eps.pdf',format="pdf", bbox_inches="tight")
plt.close()


# In[45]:


test_rewards = training_agent.test_agent(100)


# In[91]:


print("percentage of test over 200: ",len(test_rewards[test_rewards>200])/len(test_rewards))
print("Average testing rewards: ",test_rewards.mean())


# In[102]:


#plt.plot(test_rewards)
plt.style.use('_mpl-gallery')

# plot:
fig, ax = plt.subplots(figsize=(6, 4))
plt.axvline(test_rewards.mean(),label="average rewards",color='g',linestyle='--',linewidth=4)
plt.hist(test_rewards, bins=8, linewidth=0.5, edgecolor="white")

# plt.set(xlim=(-100, 300),
#        ylim=(0, 60), yticks=np.linspace(0, 56, 9))
plt.legend()
plt.ylabel('counts')
plt.xlabel('rewards')
plt.title("Rewards of 100 independent episodes")
fig.savefig('testing_100_eps.pdf',format="pdf", bbox_inches="tight")
plt.close()



# ## experiment 3: change learning rate

# In[51]:


num_test_eps = 100


# In[53]:


lrs = [1e-3,5e-3,1e-4,5e-4,6.1e-5]
rewards_lrs = []
test_rewards_lr = []
env = gym.make('LunarLander-v2')
for lr in lrs:
    training_agent = Training_agent(env,lr_optim=lr)
    start = time.time()
    training_agent.train_agent(num_eps,64)
    print ((time.time()-start)/60.)
    rewards_lrs.append(training_agent.rewards)
    
    test_rewards = training_agent.test_agent(num_test_eps)
    test_rewards_lr.append(test_rewards)
    #print("Average testing rewards: ",test_rewards.mean())
    


# In[56]:


moving_average_period = 50
x = np.arange(moving_average_period,num_eps+1)
fig = plt.figure(figsize=(6, 4))
for i in range(len(lrs)):
    moving_avg = moving_average(rewards_lrs[i],n=moving_average_period)
    plt.plot(x,moving_avg,label='lr = {}'.format(lrs[i]))
plt.xlabel('number of episodes')
plt.ylabel('rewards')
plt.title('50 epsidoes moving average for different learning rate')
plt.legend()
plt.savefig('lr_exp.pdf',format="pdf", bbox_inches="tight")
plt.close()

# In[100]:


# plot:
fig, ax = plt.subplots(figsize=(5, 5))
means = np.zeros(len(test_rewards_lr))
labels = ['1e-3','5e-3','1e-4','5e-4','1e-5']
for i in range(len(test_rewards_lr)):
    means[i] = test_rewards_lr[i].mean()

ax.bar(labels, means, width=0.9, edgecolor="white", linewidth=0.7)


# plt.set(xlim=(-100, 300),
#        ylim=(0, 60), yticks=np.linspace(0, 56, 9))
plt.title('Average rewards of 100 episodes for different learning rates')
plt.ylabel('Average rewards')
plt.xlabel('learning rate of DQN agent')
plt.savefig('lr_exp_testing.pdf',format="pdf", bbox_inches="tight")
plt.close()


# ## Exeperiment 4: change the discount 

# In[58]:


discounts = [0.8,0.9,0.99,0.999]
rewards_discounts = []
test_rewards_discount = []
env = gym.make('LunarLander-v2')
for discount in discounts:
    training_agent = Training_agent(env,discount=discount)
    start = time.time()
    training_agent.train_agent(num_eps,64)
    print ((time.time()-start)/60.)
    rewards_discounts.append(training_agent.rewards)
    
    test_rewards = training_agent.test_agent(num_test_eps)
    test_rewards_discount.append(test_rewards)
    #print("Average testing rewards: ",test_rewards.mean())
    


# In[59]:


moving_average_period = 50
x = np.arange(moving_average_period,num_eps+1)
fig = plt.figure(figsize=(6, 4))
for i in range(len(discounts)):
    moving_avg = moving_average(rewards_discounts[i],n=moving_average_period)
    plt.plot(x,moving_avg,label='dicount rate = {}'.format(discounts[i]))
plt.xlabel('number of episodes')
plt.ylabel('rewards')
plt.title('50 epsidoes moving average for different discount factor')
plt.legend()
plt.savefig('discount_exp.pdf',format="pdf", bbox_inches="tight")
plt.close()


# In[99]:


# plot:
fig, ax = plt.subplots(figsize=(5, 5))
means = np.zeros(len(test_rewards_discount))
labels = ['0.8','0.9','0.99','0.999']
for i in range(len(test_rewards_discount)):
    means[i] = test_rewards_discount[i].mean()

ax.bar(labels, means, width=0.8, edgecolor="white", linewidth=0.7)


# plt.set(xlim=(-100, 300),
#        ylim=(0, 60), yticks=np.linspace(0, 56, 9))
plt.title('Average rewards of 100 episodes for different dicount rates')
plt.ylabel('Average rewards')
plt.xlabel('Discount rate for rewards')
plt.savefig('discount_exp_testing.pdf',format="pdf", bbox_inches="tight")
plt.close()


# ## Exeperiment 5: change the decay rate

# In[62]:


decay_rates = [0.9,0.95,0.99,0.999]
rewards_decay = []
test_rewards_decay = []
#num_test_eps = 50
env = gym.make('LunarLander-v2')
for decay in decay_rates:
    training_agent = Training_agent(env,decay_rate=decay)
    start = time.time()
    training_agent.train_agent(num_eps,64)
    print ((time.time()-start)/60.)
    rewards_decay.append(training_agent.rewards)
    
    test_rewards = training_agent.test_agent(num_test_eps)
    test_rewards_decay.append(test_rewards)
    #print("Average testing rewards: ",test_rewards.mean())
    

# In[84]:


moving_average_period = 50
x = np.arange(moving_average_period,num_eps+1)
fig = plt.figure(figsize=(6, 4))
for i in range(len(decay_rates)):
    moving_avg = moving_average(rewards_decay[i],n=moving_average_period)
    plt.plot(x,moving_avg,label='decay rate = {}'.format(decay_rates[i]))

plt.xlabel('number of episodes')
plt.ylabel('rewards')
plt.title('50 epsidoes moving average for different decay rates')
plt.legend()
fig.savefig('decay_exp.pdf',format="pdf", bbox_inches="tight")
plt.close()


# In[108]:


# plot:
fig, ax = plt.subplots(figsize=(5, 5))
means = np.zeros(len(test_rewards_decay))
labels = ['0.9','0.95','0.99','0.999']
for i in range(len(test_rewards_decay)):
    means[i] = test_rewards_decay[i].mean()

ax.bar(labels, means, width=0.8, edgecolor="white", linewidth=0.7)


# plt.set(xlim=(-100, 300),
#        ylim=(0, 60), yticks=np.linspace(0, 56, 9))
plt.title('Average rewards of 100 episodes for different decay rates')
plt.ylabel('Average rewards')
plt.xlabel('decay rate for epsilon greedy')
plt.savefig('decay_exp_testing.pdf',format="pdf", bbox_inches="tight")
plt.close()

# ## experiment 5: Different value of memory size

# In[65]:


memory_sizes = [1000,5000,10000,15000]
rewards_memory_sizes = []
test_rewards_memory_sizes = []
env = gym.make('LunarLander-v2')
#num_test_eps = 50
for max_memory in memory_sizes:
    training_agent = Training_agent(env,memory_max_size =max_memory )
    start = time.time()
    training_agent.train_agent(num_eps,64)
    print ((time.time()-start)/60.)
    rewards_memory_sizes.append(training_agent.rewards)
    
    test_rewards = training_agent.test_agent(num_test_eps)
    test_rewards_memory_sizes.append(test_rewards)
    #print("Average testing rewards: ",test_rewards.mean())


# In[69]:


moving_average_period = 50
x = np.arange(moving_average_period,num_eps+1)
fig = plt.figure(figsize=(6, 4))
for i in range(len(memory_sizes)):
    moving_avg = moving_average(rewards_memory_sizes[i],n=moving_average_period)
    plt.plot(x,moving_avg,label='Memory max size = {}'.format(memory_sizes[i]))
plt.xlabel('number of episodes')
plt.ylabel('rewards')
plt.title('50 epsidoes moving average for different Replay memory size')
plt.legend()
#plt.show()
plt.savefig('memory_exp.pdf',format="pdf", bbox_inches="tight")
plt.close()

# In[110]:


# plot:
fig, ax = plt.subplots(figsize=(5, 5))
means = np.zeros(len(test_rewards_memory_sizes))
labels = ['1000','5000','10000','15000']
for i in range(len(test_rewards_decay)):
    means[i] = test_rewards_memory_sizes[i].mean()

ax.bar(labels, means, width=0.7, edgecolor="white", linewidth=0.7)


# plt.set(xlim=(-100, 300),
#        ylim=(0, 60), yticks=np.linspace(0, 56, 9))
plt.title('Average rewards of 100 episodes for different reply memory sizes')
plt.ylabel('rewards')
plt.xlabel('reply memory size')
plt.savefig('memory_exp_testing.pdf',format="pdf", bbox_inches="tight")
plt.close()