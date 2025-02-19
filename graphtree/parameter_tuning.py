#%%
import numpy as np
from graphtree import graphtree
from starboost import BoostingClassifier
from tests import Gnp, Gnp2, GnpMax, BA_vs_Watts_Strogatz, BAmax, BAone, GnpMaxFeature, Gnp1Q,Gnp2Q
from sklearn.metrics import roc_auc_score
import wandb
from joblib import Parallel, delayed
from random import shuffle

#%%

np.random.seed(seed=1133)

#%% 
#data_generator = Gnp(10, 0.3, 10, 0.2)
#data_generator = Gnp2(p=0.5, max_size=30)
#data_generator = GnpMax()
#data_generator = BA_vs_Watts_Strogatz()
#data_generator = BAmax()
#data_generator = BAone()
#data_generator = GnpMaxFeature()
data_generator = Gnp2Q()
X_train,y_train = data_generator.get_sample(1000)
X_test,y_test = data_generator.get_sample(1000)


# %%
def test(model, X, y):
    preds = model.predict(X).flatten()
    L2 = np.mean((preds - y) ** 2) 
    auc = roc_auc_score(y, preds)

    return(L2, auc)

# %%

def run(max_attention_depth, graph_depth, train_size, n_estimators, learning_rate, leafs, wb):
    gbgta = BoostingClassifier( \
        init_estimator=graphtree(max_attention_depth=max_attention_depth, graph_depths=list(range(0, graph_depth + 1)), max_number_of_leafs=leafs), \
        base_estimator=graphtree(max_attention_depth=max_attention_depth, graph_depths=list(range(0, graph_depth + 1)), max_number_of_leafs=leafs), \
        n_estimators = n_estimators,\
        learning_rate = learning_rate)

    y = np.array(y_train[:train_size])
    y = y.flatten()
    gbgta.fit(X_train[:train_size], y)
    (L2_train, auc_train) = test(gbgta, X_train, y_train)
    print("Train: L2 error %5f auc %5f" % (L2_train,auc_train))
    (L2_test, auc_test) = test(gbgta, X_test, y_test)
    print("Test: L2 error %5f auc %5f" % (L2_test,auc_test))
    wb.log({"problem": data_generator.name, "train-size": train_size, 
                "attention-depth": max_attention_depth,
                "graph-depth": graph_depth,
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "L2-train": L2_train,
                "AUC-train": auc_train,
                "L2-test": L2_test,
                "AUC_test": auc_test,
                "leafs": leafs})

#%%
def do_curve_for_depth_attention_combination(attention, graph_depth, n_estimators, learning_rate, leafs):
    wb = wandb.init(project='gta', reinit=True, entity='mlwell-rgb', group="GTA_parallel_sweep_ps_2")
    for train_size in [1000]:
        run(attention, graph_depth, train_size, n_estimators, learning_rate, leafs, wb)
    wandb.join()
    return(True)

#%%
combinations = []
l_estimators = [20, 50, 100]
l_rates = [0.1, 0.07, 0.2, 0.01, 0.3]
l_leafs = [10, 25, 50]

for e in l_estimators:
    for r in l_rates:
        for l in l_leafs:
            combinations.append((e,r,l))

shuffle(combinations)
jobs = len(combinations)
if (jobs > 16): 
    jobs = 16

Parallel(n_jobs=jobs)(delayed(do_curve_for_depth_attention_combination)(1, 1, e, r, l) for (e, r, l) in combinations )

# %%
