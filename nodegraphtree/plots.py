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

    ax = sns.lineplot(x='train-size', y='test-auc', hue='model', #alpha  = 0.4,
                    data=exp_history)
    leg = ax.legend(loc="lower right").get_lines()
    styles = ['solid', '--', ':']
    colors = ['r', 'c', 'y']
    for i in range(0, 9):
        ax.lines[i].set_linestyle(styles[int(i % 3)])
        leg[i].set_linestyle(styles[int(i % 3)])
        ax.lines[i].set_color(colors[int(i/3)])
        leg[i].set_color(colors[int(i/3)])
    ax.set(ylabel='AUC-test', title=exp)
    plt.grid(axis='y')
    plt.savefig(exp + '.png')
    plt.show()
# %%
