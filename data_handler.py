import numpy as np
import sys
from sklearn.decomposition import PCA
import csv
import os
from os.path import join, isfile

class DataHandler:

    def __init__(self):
        self.path = os.getcwd() #Get CWD


    def fetch_data(self, filename):

        #Read parameters from parameters.csv into [[]]
        # file = [f for f in os.listdir(path+filename) if isfile(join(path+filename, f))]

        ################## FETCH DATA AND PARAMETERS ##################
        all_params = self.read_csv(filename) #Read all lines from file
        Xs, Ys, data_params, train_params = self.fetch(all_params) #Handle lines, fetch data and parameters
        ###############################################################

        ######### CALC NUMBER OF DATA POINTS AND REG TASKS ############
        num_reg_tasks = len(Ys)-1 #Get number of regression tasks in data

        #Get final shapes of all input data
        shapes = []
        for X in Xs:
            print(X.shape)
            shapes.append(X.shape[-1])
        ###############################################################

        return Xs, Ys, train_params, self.num_data, num_reg_tasks, shapes, data_params['n_steps']



    def fetch(self, all_params):
        #Loop through each row of parameters
        for x in all_params:
            if x[0] == "#X,": #If params are X files
                Xs = self.handle_X(x[1:])
                self.num_data = Xs[list(Xs.keys())[0]].shape[0] #Get number of data points

            if x[0] == "#Y,": #If params are Y files
                Ys = self.handle_Y(x[1:])

            if x[0] == "#P,": #If params are data parameters
                data_params = self.handle_P(x[1:])

            if x[0] == "#T,": #If params are training parameters
                train_params = self.handle_T(x[1:])

        ########### ADJUST DATA ACCORDING TO PARAMS FETCHED ###########
        #Look for and properly format Y_class data
        for y in Ys.keys():
            if 'class' in y:
                Ys[y] = Ys[y].reshape((self.num_data, 4))

        #Perform PCA on data specified in #P
        for p in data_params.keys():
            for key in Xs.keys():
                if p in key:
                    Xs[key] = self.perform_pca(Xs[key], data_params[p]) #Perform PCA

        #Reduce input data to number of timesteps required
        X_final = [self.reduce_timestepsX(X, data_params['n_steps']) for X in list(Xs.values())[:-1]] + [list(Xs.values())[-1]] #Add background data in on the end
        Y_final = [list(Ys.values())[0]] + [self.reduce_timestepsY(Y, data_params['n_steps']) for Y in list(Ys.values())[1:]]
        ###############################################################


        return X_final, Y_final, data_params, train_params

    def handle_X(self, x):
        Xs = {i:np.load(self.path+'\\data\\'+i.strip(',')) for i in x} #Load into dict {}
        return Xs

    def handle_Y(self, x):
        Ys = {i:np.load(self.path+'\\data\\'+i.strip(',')) for i in x} #Load into dict {}
        return Ys

    def handle_P(self, x):
        params = {i.split(':')[0]:int(i.split(':')[1].strip(',')) for i in x}
        return params

    def handle_T(self, x):
        train_params = {i.split(':')[0]:int(i.split(':')[1].strip(',')) for i in x}
        return train_params

    def read_csv(self, dir):
        all_params = []
        with open(self.path+'\\parameters\\'+dir, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for i, row in enumerate(reader):
                all_params.append(row)
        return all_params

    def perform_pca(self, data, n_comp = 100):
        #Take numpy data as input and perform PCA on it, keeping n best components
        #Input: Numpy 3d array, int
        #Output: Numpy 3d array
        print('\nPerforming PCA')
        print('Original data shape: ')
        print(data.shape)
        pca = PCA(n_components=n_comp)
        n_steps = data.shape[1]
        reshaped_data = data.reshape((self.num_data*n_steps,data.shape[-1]))
        pca.fit(reshaped_data)

        transformed_data = np.empty([self.num_data,n_steps,n_comp])

        for i in range(len(data)):
             transformed_data[i]=pca.transform(data[i])
        print('\nFinal data shape: ')
        print(transformed_data.shape, '\n')
        return transformed_data

    def reduce_timestepsX(self, X, n_steps):
        #Reduce total time steps in the data to that required
        #Input: Numpy 3d array, int
        #Output: Numpy 3d array
        transformed_X = np.empty([self.num_data,n_steps,X.shape[-1]])
        transformed_X = X[:,:n_steps,:]
        return transformed_X

    def reduce_timestepsY(self, Y, n_steps):
        #Reduce total time steps in the data to that required
        #Input: Numpy 3d array, int
        #Output: Numpy 3d array
        transformed_Y = np.empty([self.num_data,1])
        transformed_Y = Y[:,n_steps,:]
        return transformed_Y
