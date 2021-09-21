#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

history = pd.read_csv("project.csv")


#%%
experiments = history['problem'].unique().tolist()
experiments
# %%
for exp in experiments:
    print(exp)
    exp_history = history[history['problem'] == exp]
    exp_history['model'] = 'a' + exp_history['attention-depth'].astype(str) + ' d' + exp_history['graph-depth'] .astype(str)

    ax = sns.lineplot(x='train-size', y='AUC_test', hue='model', 
                    data=exp_history)
    leg = ax.legend().get_lines()
    styles = ['solid', '--', ':']
    colors = ['r', 'c', 'y']
    for i in range(0, 9):
        ax.lines[i].set_linestyle(styles[int(i % 3)])
        leg[i].set_linestyle(styles[int(i % 3)])
        ax.lines[i].set_color(colors[int(i/3)])
        leg[i].set_color(colors[int(i/3)])
    ax.set(ylabel='AUC-test', title=exp)
    plt.savefig(exp + '.png')
    plt.show()
# %%
