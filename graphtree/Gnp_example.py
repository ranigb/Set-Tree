#%%
import numpy as np
from graphtree import graphtree
from starboost import BoostingClassifier
from tests import Gnp, Gnp2, GnpMax, BA_vs_Watts_Strogatz, BAmax, BAone, GnpMaxFeature
from sklearn.metrics import roc_auc_score
import wandb
wandb.init(project='gta', entity='mlwell-rgb')

#%%

np.random.seed(seed=1133)

#%% 
#data_generator = Gnp(10, 0.3, 10, 0.2)
#data_generator = Gnp2(p=0.5, max_size=30)
#data_generator = GnpMax()
#data_generator = BA_vs_Watts_Strogatz()
#data_generator = BAmax()
#data_generator = BAone()
data_generator = GnpMaxFeature()
X_train,y_train = data_generator.get_sample(1000)
X_test,y_test = data_generator.get_sample(1000)


# %%
def test(model, X, y):
    preds = model.predict(X).flatten()
    L2 = np.mean((preds - y) ** 2) 
    auc = roc_auc_score(y, preds)

    return(L2, auc)

# %%

def run(max_attention_depth, graph_depth, train_size):
    n_estimators = 30
    learning_rate = 0.1
    gbgta = BoostingClassifier( \
        init_estimator=graphtree(max_attention_depth=max_attention_depth, graph_depths=list(range(0, graph_depth + 1))), \
        base_estimator=graphtree(max_attention_depth=max_attention_depth, graph_depths=list(range(0, graph_depth + 1))), \
        n_estimators = n_estimators,\
        learning_rate = learning_rate)

    y = np.array(y_train[:train_size])
    y = y.flatten()
    gbgta.fit(X_train[:train_size], y)
    (L2_train, auc_train) = test(gbgta, X_train, y_train)
    print("Train: L2 error %5f auc %5f" % (L2_train,auc_train))
    (L2_test, auc_test) = test(gbgta, X_test, y_test)
    print("Test: L2 error %5f auc %5f" % (L2_test,auc_test))
    wandb.log({"problem": data_generator.name, "train-size": train_size, 
                "attention-depth": max_attention_depth,
                "graph-depth": graph_depth,
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "L2-train": L2_train,
                "AUC-train": auc_train,
                "L2-test": L2_test,
                "AUC_test": auc_test})
for attention in range(0,3):
    for graph_depth in range(0,3):
        wb = wandb.init(reinit=True)
        for train_size in [10,20,50,100, 200, 500, 1000]:
            print("*** Training with attention=%d graph-depth=%d train_size=%d***" % (attention, graph_depth, train_size))
            run(attention, graph_depth, train_size)
        wb.finish()

# %%
