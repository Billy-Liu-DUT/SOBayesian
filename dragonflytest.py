import numpy as np
from argparse import Namespace
from dragonfly import load_config
from dragonfly.exd import domains
from dragonfly.exd.experiment_caller import CPFunctionCaller, CPMultiFunctionCaller
from dragonfly.opt.gp_bandit import CPGPBandit
from dragonfly.opt.multiobjective_gp_bandit import CPMultiObjectiveGPBandit
from dragonfly.exd.worker_manager import SyntheticWorkerManager
import pandas as pd
import math
import csv
import matplotlib.pyplot as plt # for plotting

snar_data = pd.read_csv(r'C:\Users\Billy\Desktop\算法\NGBoost-master (1)\NGBoost-master\random_points_yield.csv')
n_data = len(snar_data)

# Reaction conditions evaluated
time = snar_data['time'].values
temp = snar_data['temperature '].values

# Objective values
yields = snar_data['Yield (%)'].values

# Convert to data format required by algorithm
X = [] # input values
Y = [] # objective values
for i in range(n_data):
    x = [ temp[i], time[i]]
    y = [yields[i]]
    X.append(x)
    Y.append(y)

# Define variables (same order as in 'X' object)
variables = [   {'name': 'temperature', 'type': 'float',  'min': 25, 'max': 75},
                {'name': 'time', 'type': 'float',  'min': 10, 'max': 120},

            ]

# Create domain from variables
config_params = {'domain': variables}
config = load_config(config_params)

# Set up Dragonfly optimizer object
num_init = 1 # specify any value
options = Namespace(
    gpb_hp_tune_criterion = 'ml',  # Criterion for tuning GP hyperparameters.
                                   # Options: 'ml' (maximum likelihood), 'post_sampling' (posterior sampling).
    )

func_caller = CPFunctionCaller(None, config.domain, domain_orderings=config.domain_orderings)
wm = SyntheticWorkerManager(1)

opt = CPGPBandit(func_caller, 'default', ask_tell_mode=True, options=options)
opt.worker_manager = None
opt._set_up()


# ------------ Regress algorithm's Gaussian process (GP) model to data ------------

opt.initialise() # this generates initialization points
init_expts = opt.ask(num_init) # extract and discard all algorithm-generated initial points (will not be used)

# Return values from dataset to algorithm
for i in range(n_data):
    x = X[i]
    y = Y[i][0]
    opt.tell([(x, y)]) # return result to algorithm
    opt.step_idx += 1 # increment experiment number
    print("expt #:", opt.step_idx, ", x:", x, )
# Update GP model using results
opt._build_new_model() # key line! update model using prior results
opt._set_next_gp() # key line! set next GP
print("expt #:", opt.step_idx, ", x:", x,)
# Extract GP models
gp_yield = opt.gp


next_points = opt.ask(1)

# 输出这个点
for point in next_points:
    print("next point", point)

# Code for obtaining model prediction given an input value x
x_raw = list(next_points[0])
x_input = [opt.func_caller.get_processed_domain_point_from_raw(x)]



mu, stdev = gp_yield.eval(x_input, uncert_form='std')
print(mu, stdev)
