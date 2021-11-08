#%%
import numpy as np
from graphtree import graphtree
from starboost import BoostingClassifier
from tests import *
from sklearn.metrics import roc_auc_score
import wandb
import sys
#%%
no_debug = (sys.gettrace() == None)
#no_debug = False

#%%

np.random.seed(seed=1133)

#%% 
#data_generator = Gnp_sign_neighbor(0.2)

#data_generator = Gnp_sign_red_neighbor(0.02)
data_generator = Gnp_sign_red_blue_neighbor(0.02)
# %%
def test(model, X, y):
    y = y.A1
    preds = model.predict(X).flatten()
    L2 = np.mean((preds - y) ** 2) 
    auc = roc_auc_score(y, preds)

    return(L2, auc)

# %%

def run(max_attention_depth, graph_depth, train_size, graph, y, n_estimators, learning_rate):
    train_set = [x for x in range(0, train_size)]
    test_set = [x for x in range(int(graph_size/2) + 1, graph_size)]

    gbgta = BoostingClassifier( \
        init_estimator=graphtree(graph=graph, max_attention_depth=max_attention_depth, graph_depths=list(range(0, graph_depth + 1))), \
        base_estimator=graphtree(graph=graph, max_attention_depth=max_attention_depth, graph_depths=list(range(0, graph_depth + 1))), \
        n_estimators = n_estimators,\
        learning_rate = learning_rate)

    gbgta.fit(train_set, y[train_set])
    (L2_train, auc_train) = test(gbgta, train_set, y[train_set])
    print("Train: L2 error %5f auc %5f" % (L2_train,auc_train))
    (L2_test, auc_test) = test(gbgta, test_set, y[test_set])
    print("Test: L2 error %5f auc %5f" % (L2_test,auc_test))
    if (no_debug):
        wandb.log(({"train-size": train_size,
                    "train-auc": auc_train,
                    "train-l2": L2_train,
                    "test-auc": auc_test,
                    "test-l2": L2_test
                    }))

graph_size = 1000
n_estimators = 50
learning_rate = 0.1
graph, y, train_set, test_set = data_generator.get_graph(graph_size)

for attention in range(0,3):
    for graph_depth in range(0,3):
        if (no_debug):
            wb = wandb.init(project='gta-node', reinit=True, 
                config=
                {"problem": data_generator.name,  
                 "graph-size": graph_size,
                 "attention-depth": attention,
                 "graph-depth": graph_depth,
                 "n_estimators": n_estimators,
                 "learning_rate": learning_rate})


        for train_size in [20, 50, 100, 200, 350, 500]:
 
            print("*** Training with attention=%d graph-depth=%d train_size=%d***" % (attention, graph_depth, train_size))
            run(attention, graph_depth, train_size, graph, y, n_estimators, learning_rate)
        if (no_debug):
            wb.finish()


# %%

def verify():
    graph, y, train_set, test_set = data_generator.get_graph(100)
    v = graph.propagate(1, list(range(0, graph.get_number_of_nodes())))
    y = np.squeeze(np.asarray(y))
    gta = graphtree(graph=graph, max_attention_depth=1, graph_depths=list(range(0, 3)))
    train_set = np.array(train_set)
    gta.fit(train_set, y)
    lrnr = gta.tree_learner_
    mat, att = lrnr.get_feature_matrix()
    print(mat)
    print("*****")
    print(v)



#verify()
# %%
