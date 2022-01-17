import numpy as np
from data_handler import DataHandler

h = DataHandler()
Xs, Ys, train_params, num_data, num_reg_tasks, shapes, n_steps = h.fetch_data('parameters.csv')

print(num_data)
