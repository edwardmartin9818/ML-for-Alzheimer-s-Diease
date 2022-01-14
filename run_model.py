from model import Ensemble_Model
from data_handler import DataHandler
import numpy as np
import sys
import csv
import os
from os.path import join, isfile


#Read parameters from parameters.csv into [[]]
path = os.getcwd()
path_param = '\\parameters\\'
onlyfiles = [f for f in os.listdir(path+path_param) if isfile(join(path+path_param, f))]
print(onlyfiles)
for file in onlyfiles:

    handler = DataHandler()
    Xs, Ys, train_params, num_data, num_reg_tasks, shapes, n_steps = handler.fetch_data(file)

    final_results = {}
    historys = []
    states = []

    print('Creating model...')
    model = Ensemble_Model(num_reg_tasks, input_shapes=shapes, steps = n_steps)
    print('Model created.\n')

    print('Beginning training...')
    history, state = model.train_model(Xs, Ys, epochs = train_params['epochs'], batch_size = train_params['batch'])
    print('Train complete.\n')

    # print(history.history.keys())
    # print(history.history['output_diagnosis_accuracy'])

    historys.append(history)
    states.append(state)


    for key in history.history.keys():
        temp = []
        for h in historys:
            temp.append(h.history[key])
        mean = np.mean(np.asarray(temp),axis=0)
        final_results[key] = mean
        print(final_results[key])

    with open('results\\r_'+file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', )
        writer.writerow('Used params from: '+ file)
        for row in all_params:
            writer.writerow(row)
        for key in final_results:
            to_write = []
            for i in range(len(final_results[key])):
                to_write.append(str(final_results[key][i]))
                print('To write:', to_write)
            writer.writerow([key+':'] + to_write)
