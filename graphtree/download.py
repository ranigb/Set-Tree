#%%
import wandb
api = wandb.Api()

runs = api.runs("mlwell-rgb/gta")
summary_list = [] 
config_list = [] 
name_list = [] 
history = []
for run in runs: 
    if (len(history) == 0):
        history = run.history().copy()
    else:
        history = history.append(run.history())

import pandas as pd 
history.to_csv("projectQ2.csv")
