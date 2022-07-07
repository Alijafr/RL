#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ray


# In[2]:


from rldm.utils import system_tools as st
n_cpus, n_gpus = st.get_cpu_gpu_count()
debug = False
ray.init(num_cpus=n_cpus, num_gpus=n_gpus, local_mode=debug)


# In[21]:


from ray import tune


# ## register the env

# In[14]:


from gfootball import env as fe
from rldm.utils import football_tools as ft
from ray.tune.registry import register_env
import numpy as np
num_players = 3
shared_policy = False
n_policies = 1 if shared_policy else num_players - 1 # hard-coding
env_name = ft.n_players_to_env_name(num_players, True)
register_env(env_name, lambda _: ft.RllibGFootball(env_name=env_name))


# ## configuration for the experiment 

# In[10]:





# In[19]:


obs_space, act_space = ft.get_obs_act_space(env_name)

def gen_policy(idx):
    return (None, obs_space[f'player_{idx}'], act_space[f'player_{idx}'], {})

policies = {
        'agent_{}'.format(idx): gen_policy(idx) for idx in range(n_policies)
    }

policy_ids = list(policies.keys())

policy_mapping_fn = lambda agent_id, episode, **kwargs:         policy_ids[0 if len(policy_ids) == 1 else int(agent_id.split('_')[1])]
#in case of using a indiviual policy
default_multiagent = {
        'policies': policies,
        'policy_mapping_fn': policy_mapping_fn,
    }
#in case of using a shared policy 
shared_policy = {'agent_0': gen_policy(0)}
shared_policy_mapping_fn = lambda agent_id, episode, **kwargs: 'agent_0'
shared_multiagent = {
    'policies': shared_policy,
    'policy_mapping_fn': shared_policy_mapping_fn,
}


# ## Qmix

# In[35]:


from rldm.utils.collection_tools import deep_merge
import random
use_tune_config = True
# config = {"env":env_name,
#           "double_q": False,
#             "rollout_fragment_length": 100,
#           "train_batch_size": 2_800,
#            "num_gpus":n_gpus,
#            "num_workers":n_cpus-1,
#            "multiagent": shared_multiagent}

# if use_tune_config:
#     tune_config = {
#         "train_batch_size": tune.randint(32,64),
#         "lr": tune.uniform(0.00005, 0.0005),
#         # === Model ===
#         "model": {
#             "lstm_cell_size": tune.randint(45,80)
#         },
#         # "multiagent": tune.choice([default_multiagent, shared_multiagent]),
#     }
#     config = deep_merge(config, tune_config)




# ## add a scheduler to terminate any bad trial

# In[46]:


# from ray.tune.schedulers import ASHAScheduler
# use_callbacks = False
# if use_callbacks:
#     config['callbacks'] = ft.FootballCallbacks

use_scheduler = False
n_timesteps =20_000_000
# scheduler = None
stop = {
    "timesteps_total": n_timesteps,
}
# if use_scheduler: 
#     scheduler = ASHAScheduler(
#         time_attr='timesteps_total',
#         metric='episode_reward_mean',
#         mode='max',
#         max_t=n_timesteps,
#         grace_period=int(n_timesteps*0.10),
#         reduction_factor=3,
#         brackets=1)
#     stop = None


# In[48]:





# In[47]:


import os 
filename_stem = os.path.basename(__file__).split(".")[0]
policy_type = 'search' if use_tune_config else     'shared' if n_policies == 1 else 'independent'
scheduler_type = 'asha' if use_scheduler else 'fifo'
config_type = 'tune' if use_tune_config else 'fixed'
experiment_name =f"{filename_stem}_{env_name}_{policy_type}_{n_timesteps}_{scheduler_type}_{config_type}"
script_dir = os.path.dirname(os.path.realpath(__file__))
local_dir = os.path.join(script_dir, '..', '..', 'logs')
print(experiment_name)


# In[ ]:


n_samples = 8


a = tune.run(
    "QMIX",
    stop=stop,
    config={
        "env":env_name ,
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001])
    },
    checkpoint_freq=100,
    checkpoint_at_end=True,
    local_dir=local_dir,
    num_samples=2,
    reuse_actors=False,
    fail_fast=True,
    max_failures=0
)
# In[36]:


# a = tune.run("QMIX",
#         name=experiment_name,
#         reuse_actors=False,
#         scheduler=scheduler,
#         raise_on_failed_trial=True,
#         fail_fast=True,
#         max_failures=0,
#         num_samples=n_samples,
#         stop=stop,
#         checkpoint_freq=100,
#         checkpoint_at_end=True,
#         local_dir=local_dir,
#         config=config,
#         verbose=1 if not debug else 3
#         )


# In[ ]:


checkpoint_path = a.get_best_checkpoint(a.get_best_trial("episode_reward_mean", "max"), "episode_reward_mean", "max")
print('Best checkpoint found:', checkpoint_path)
ray.shutdown()

