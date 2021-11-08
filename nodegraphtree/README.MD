### install starboost and introduce the following changes:

### line 124 of boosting.py in starboost should be changed to:

###   y_pred[:, i] += self.learning_rate * direction[:, i]   

### The same fix should be applied in line 179               

### /usr/local/lib/python3.8/dist-packages/starboost/        

download.py    - download data from wandb to allow the generation of nice graphs

graphtree.py  - holds a GTA

tests.py      - different experiments settings

tree_node_learner.py - learns a tree
                 
graph_data.py  - holds graph data

plots.py      - plots nice graphs

run.py       - runs experiments

tree_node.py - a single tree node
