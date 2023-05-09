#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install stable-baselines3[extra]


# In[ ]:


#!pip install gym[all]


# In[1]:


#!pip install pyglet==1.5.27


# In[14]:


import gym 
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# In[3]:


environment_name = "CartPole-v0"
env = gym.make(environment_name)


# In[4]:


environment_name


# In[10]:


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
    print('Episode:{} Score:{}'.format(episode, score))
env.close()


# In[11]:


env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)


# In[12]:


model.learn(total_timesteps=20000)


# # Saving, Loading and running model

# In[15]:


PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')


# In[16]:


model.save(PPO_path)


# In[17]:


del model


# In[19]:


model = PPO.load(PPO_path, env=env)


# In[20]:


from stable_baselines3.common.evaluation import evaluate_policy
evaluate_policy(model, env, n_eval_episodes=10, render=True)


# In[21]:


env.close()


# # Test1

# In[22]:


episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))


# In[23]:


env.close()


# # Test2

# In[24]:


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done: 
        print('info', info)
        break


# In[25]:


env.close()

