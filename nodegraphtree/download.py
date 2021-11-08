#%%
import wandb
api = wandb.Api()

runs = api.runs("mlwell-rgb/gta-node")
summary_list = [] 
config_list = [] 
name_list = [] 
history = []
for run in runs: 
    r = run.history().copy()
    conf = run.config
    for c in conf:
        r[c] = conf[c]
    if (len(history) == 0):
        history = r
    else:
        history = history.append(r)

import pandas as pd 
history.to_csv("project.csv")

# %%
