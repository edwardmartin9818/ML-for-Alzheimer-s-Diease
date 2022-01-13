from final_model import Ensemble_Model
# import numpy as np
# import sys
# from sklearn.decomposition import PCA
# import csv
# import os
# from os.path import join, isfile

#Take numpy data as input and perform PCA on it, keeping n best components
#Input: Numpy 3d array, int
#Output: Numpy 3d array
def perform_pca(data, n_comp = 100):
    print('\nPerforming PCA')
    print('Original data shape: ')
    print(data.shape)
    pca = PCA(n_components=n_comp)
    n_steps = data.shape[1]
    reshaped_data = data.reshape((num_data*n_steps,data.shape[-1]))
    pca.fit(reshaped_data)

    transformed_data = np.empty([num_data,n_steps,n_comp])

    for i in range(len(data)):
         transformed_data[i]=pca.transform(data[i])
    print('\nFinal data shape: ')
    print(transformed_data.shape, '\n')
    return transformed_data

#Reduce total time steps in the data to that required
#Input: Numpy 3d array, int
#Output: Numpy 3d array
def reduce_timestepsX(X, n_steps):
    transformed_X = np.empty([num_data,n_steps,X.shape[-1]])
    transformed_X = X[:,:n_steps,:]
    return transformed_X

#Reduce total time steps in the data to that required
#Input: Numpy 3d array, int
#Output: Numpy 3d array
def reduce_timestepsY(Y, n_steps):
    transformed_Y = np.empty([num_data,1])
    transformed_Y = Y[:,n_steps,:]
    return transformed_Y

#Read parameters from parameters.csv into [[]]
path = os.getcwd()
path_param = '\\parameters\\'
onlyfiles = [f for f in os.listdir(path+path_param) if isfile(join(path+path_param, f))]
print(onlyfiles)
for file in onlyfiles:
    final_results = {}
    historys = []
    states = []
    for i in range(2):
        all_params = []
        with open(path+'\\parameters\\'+file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for i, row in enumerate(reader):
                all_params.append(row)
        ############### DATA RETREIVAL AND PREPERATION #################################
        #Loop through each row of parameters
        for x in all_params:

            #Load all files into dict as such: Xs[filename] = (numpy file read in)
            if x[0] == "#X,": #If params are X files
                Xs = {i:np.load(path+'\\data\\'+i.strip(',')) for i in x[1:]} #Load into dict {}
                num_data = Xs[list(Xs.keys())[0]].shape[0]
            #Load all files into dict as such: Ys[filename] = (numpy file read in)
            if x[0] == "#Y,": #If params are Y files
                print(x)
                Ys = {i:np.load(path+'\\data\\'+i.strip(',')) for i in x[1:]} #Load into dict {}
                #Look for and properly format Y_class data
                for y in Ys.keys():
                    if 'class' in y:
                        Ys[y] = Ys[y].reshape((num_data, 4))

                num_reg_tasks = len(Ys)-1

            #Load all files into dict as such: data_params[param] = (some number)
            if x[0] == "#P,": #If params are data parameters
                data_params = {i.split(':')[0]:int(i.split(':')[1].strip(',')) for i in x[1:]}
                #Loop through all params to make adjustments
                for p in data_params.keys():
                    for key in Xs.keys():
                        if p in key:
                            Xs[key] = perform_pca(Xs[key], data_params[p]) #Perform PCA

                #Reduce input data to number of timesteps required
                X_final = [reduce_timestepsX(X, data_params['n_steps']) for X in list(Xs.values())[:-1]] + [list(Xs.values())[-1]] #Add background data in on the end
                Y_final = [list(Ys.values())[0]] + [reduce_timestepsY(Y, data_params['n_steps']) for Y in list(Ys.values())[1:]]

            if x[0] == "#T,": #If params are training parameters
                train_params = {i.split(':')[0]:int(i.split(':')[1].strip(',')) for i in x[1:]}

        ################################################################################

        shapes = []
        for X in X_final:
            print(X.shape)
            shapes.append(X.shape[-1])

        print('Creating model...')
        model = Ensemble_Model(num_reg_tasks, input_shapes=shapes, steps = data_params['n_steps'])
        print('Model created.\n')

        print('Beginning training...')
        history, state = model.train_model(X_final, Y_final, epochs = train_params['epochs'], batch_size = train_params['batch'])
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
