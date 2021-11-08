You should install starboost and introduce the following changes:
line 124 of boosting.py in starboost should be changed to:
   y_pred[:, i] += self.learning_rate * direction[:, i]   

The same fix should be applied in line 179               
/usr/local/lib/python3.8/dist-packages/starboost/        


download.py    - downloading data from wandb to allow drawing nice graphs
graphtree.py   - represents a GTA tree      
plots.py       - plots nice graphs
run_parallel.py    - runs tests in parallel
tests.py       - defines different tasks 
tree_node_learner.py - learning a tree
graph_data.py  - holding graph data
parameter_tuning.py  - a small experiment to tune tree parameters
run.py    - runs test in serial mode
split_criteria.py  - defines the possible splits allowed in trees
tree_node.py - a single node in the tree
