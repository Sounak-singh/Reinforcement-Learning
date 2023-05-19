#!/usr/bin/env python
# coding: utf-8

# In[27]:


import gym
import os
from stable_baselines3 import PPO
from time import sleep
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# In[3]:


environment_name = "ALE/Alien-v5"
env = gym.make(environment_name)


# In[24]:


episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
        sleep(0.2)
    print('Episode:{} Score:{}'.format(episode, score))
    
#env.close()


# In[25]:


env.close()


# In[28]:


env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)


# In[29]:


model.learn(total_timesteps=20000)


# In[43]:


PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model_911')


# In[44]:


model.save(PPO_path)


# In[47]:


episodes = 10
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        #sleep(0.05)
    print('Episode:{} Score:{}'.format(episode, score))


# In[48]:


env.close()


# # Testing Cnn policy

# In[34]:


model = PPO('CnnPolicy', env, verbose = 1)


# In[42]:


model.learn(total_timesteps=5000)


# In[ ]:




