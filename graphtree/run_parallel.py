#%%
import numpy as np
from graphtree import graphtree
from starboost import BoostingClassifier
from tests import Gnp, Gnp2, GnpMax, BA_vs_Watts_Strogatz, BAmax, BAone, GnpMaxFeature, Gnp1Q,Gnp2Q
from sklearn.metrics import roc_auc_score
import wandb
from joblib import Parallel, delayed
import datetime

group =  "GTA-parallel " + str(datetime.datetime.now())
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
X_train,y_train = data_generator.get_sample(10000)
X_test,y_test = data_generator.get_sample(1000)


# %%
def test(model, X, y):
    preds = model.predict(X).flatten()
    L2 = np.mean((preds - y) ** 2) 
    auc = roc_auc_score(y, preds)

    return(L2, auc)

# %%

def run(max_attention_depth, graph_depth, train_size, wb):
    n_estimators = 100
    learning_rate = 0.2
    leafs = 25
    gbgta = BoostingClassifier( \
        init_estimator=graphtree(max_attention_depth=max_attention_depth, graph_depths=list(range(0, graph_depth + 1)),max_number_of_leafs=leafs), \
        base_estimator=graphtree(max_attention_depth=max_attention_depth, graph_depths=list(range(0, graph_depth + 1)),max_number_of_leafs=leafs), \
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
def do_curve_for_depth_attention_combination(attention, graph_depth):
    wb = wandb.init(project='gta', reinit=True, entity='mlwell-rgb', group=group)
    for train_size in [10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10]:
#    for train_size in [10,20,50,100, 200, 500, 1000, 2000, 5000, 10000]:
        print("*** Training with attention=%d graph-depth=%d train_size=%d***" % (attention, graph_depth, train_size))
        run(attention, graph_depth, train_size, wb)
    wandb.join()
    return(True)

#%%
attention_depth_combinations = []

for attention in range(0,3):
    for graph_depth in range(0,3):
        attention_depth_combinations.append((attention, graph_depth))

Parallel(n_jobs=9)(delayed(do_curve_for_depth_attention_combination)(attention, graph_depth) for (attention, graph_depth) in attention_depth_combinations )

# %%
