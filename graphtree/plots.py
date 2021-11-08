#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

history = pd.read_csv("projectQ2.csv")
gnn_sizes = [10,20,50,100,200,500,1000,2000,5000,10000]
gnn_auc = [0.4787,0.492,0.4988,0.5136,0.5824,0.6318,0.6607,0.6796,0.7286,0.8834]


#%%
experiments = history['problem'].unique().tolist()
experiments
# %%
for exp in experiments:
    print(exp)
    exp_history = history[history['problem'] == exp]
    exp_history['model'] = 'a' + exp_history['attention-depth'].astype(str) + ' d' + exp_history['graph-depth'] .astype(str)
    exp_history = exp_history.sort_values(['model'], ascending=False)
    dict_list = []
    for i in range(0, len(gnn_sizes)):
        d = {'train-size':gnn_sizes[i],'AUC_test':gnn_auc[i],'model':'GNN'}
        dict_list.append(d)
    exp_history = exp_history.append(dict_list)


    ax = sns.lineplot(x='train-size', y='AUC_test', hue='model', 
                    data=exp_history)
    leg = ax.legend().get_lines()
    styles = ['solid', '--', ':']
    colors = ['r', 'c', 'y', 'm']
    for i in range(0, 10):
        ax.lines[i].set_linestyle(styles[int(i % 3)])
        leg[i].set_linestyle(styles[int(i % 3)])
        ax.lines[i].set_color(colors[int(i/3)])
        leg[i].set_color(colors[int(i/3)])
    ax.set(ylabel='AUC-test', title=exp)
    plt.grid(axis='y')
    plt.savefig(exp + 'Q2.png')
    plt.show()
# %%
