from tensorflow import keras as k
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.layers.merge import concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import MeanAbsoluteError, Accuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import classification_report
import numpy as np
import random


class Ensemble_Model:
    def __init__(self, reg_tasks, input_shapes, steps, opt_type):
        self.all_inputs = [] #Stores all inputs for final model
        self.lstm_inputs = [] #Stores inputs for LSTM models
        self.lstm_outputs = [] #Stores outputs for LSTM models
        self.lstm_piplines = [] #Stores LSTM models
        self.model_outputs = []

        #------------------------------------------------------------------------
        #Define each LSTM pipeline, loop through each item in input shapes and
        #create pipeline for each data modality
        pipeline_names = ['npath','npysc','cog','mri','pet'] #Corresponding data modality names

        for i in range(len(input_shapes)-1):
            model = self.lstm_pipeline((steps,input_shapes[i]),pipeline_names[i]) #Create pipeline
            self.lstm_piplines.append(model) #Append to self.lstm_piplines
            self.all_inputs.append(model.input) #Append input layers of pipelines, used in defining overall model
            self.lstm_inputs.append(model.input) #Append input layers of pipelines, used in merging pipelines together
            self.lstm_outputs.append(model.output) #Append output layers of pipelines, used for merging pipelines together

        #------------------------------------------------------------------------



        #------------------------------------------------------------------------
        #Define background_data_pipeline
        self.background_pipeline = self.background_data_pipeline(input_shape = input_shapes[-1])
        self.all_inputs.append(self.background_pipeline.input)
        self.bg_output = self.background_pipeline.output
        #------------------------------------------------------------------------



        #------------------------------------------------------------------------
        #Merge all pipelines in to final latent vector
        #Define concatenating layer with outputs of ensemble layers
        merge_pipes = concatenate(self.lstm_outputs) #Merge LSTM pipelines
        #Define dense layers after final concatenate
        regularizer = l2(1e-2) #Define regularizer
        dropout_rate = 0.2 #Define dropout rate
        x = Dense(64, activation='relu', kernel_regularizer=regularizer)(merge_pipes)
        x = Dropout(dropout_rate)(x)
        x = Dense(64, activation='relu', kernel_regularizer=regularizer)(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(64, activation='relu', kernel_regularizer=regularizer)(x)
        x = Dropout(dropout_rate)(x)
        #Merge LSTM and background data pipelines
        final_merge = concatenate([x,self.bg_output])
        x = Dense(128, activation='relu', kernel_regularizer=regularizer)(final_merge)
        ensemble_output = Dropout(0.2)(x)
        #------------------------------------------------------------------------



        ### FINAL CONCATINATION COMPLETE, SPLIT INTO MULTIPLE TASKS FROM HERE ###
        #Split in to 5 tasks
        regularizer = l2(1e-3)
        dropout_rate = 0.2
        #------------------------------------------------------------------------
        #TASK1 - Diagnosis
        x = Dense(32, activation='relu', kernel_regularizer=regularizer)(ensemble_output)
        x = Dropout(dropout_rate)(x)
        x = Dense(32, activation='relu', kernel_regularizer=regularizer)(x)
        x = Dropout(dropout_rate)(x)
        output_diagnosis = Dense(4, activation='softmax', name='output_diagnosis')(x)
        self.model_outputs.append(output_diagnosis)
        #------------------------------------------------------------------------
        for i in range(reg_tasks):
            #------------------------------------------------------------------------
            #TASKS: Regression
            x = Dense(32, activation='relu', kernel_regularizer=regularizer)(ensemble_output)
            x = Dropout(dropout_rate)(x)
            x = Dense(32, activation='relu', kernel_regularizer=regularizer)(x)
            x = Dropout(dropout_rate)(x)
            reg_task = Dense(1, activation='sigmoid',name='rt_'+str(i))(x)
            self.model_outputs.append(reg_task)
            #------------------------------------------------------------------------


        #------------------------------------------------------------------------
        #Define final model
        self.model = k.Model(inputs=self.all_inputs,outputs=self.model_outputs)

        metrics = {'rt_'+str(i):MeanAbsoluteError() for i in range(reg_tasks)}
        metrics['output_diagnosis'] = 'accuracy'


        loss = {'rt_'+str(i):'mae' for i in range(reg_tasks)}
        loss['output_diagnosis'] = 'categorical_crossentropy'


        if opt_type == 'SGD':
            opt = SGD(learning_rate=0.01)
        if opt_type == 'ADAM':
            opt = Adam(learning_rate=0.005)


        self.model.compile(loss=loss, optimizer=opt, metrics=metrics)
        #------------------------------------------------------------------------
        self.save_model_image(self.model, 'Exp1_model.png')

    def lstm_pipeline(self, input_shape, name):
        #Define input layer
        inputs = k.Input(shape=input_shape,name=name+'_input')
        #Define conv1d layer
        x = Conv1D(filters=128, kernel_size=4,strides=1, activation='relu', padding='same')(inputs)
        #Define max pooling layer
        x  = MaxPooling1D(pool_size=2, strides=2,padding='same')(x) #TO DO: figure out how this is setup



        #Define 3 stacked bi-lstm layers
        lstm_dropout = 0.1
        lstm_regularizer = l2(1e-2)
        x = Bidirectional(LSTM(128, return_sequences=True, activation='tanh', dropout=lstm_dropout, kernel_regularizer=lstm_regularizer))(x)
        x = Bidirectional(LSTM(128, return_sequences=True, activation='tanh', dropout=lstm_dropout, kernel_regularizer=lstm_regularizer))(x)
        x = Bidirectional(LSTM(128, return_sequences=False, activation='tanh', dropout=lstm_dropout, kernel_regularizer=lstm_regularizer))(x)

        #Define 2 stacked Dense layers followed by Dropout layers
        dense_regularizer = l2(1e-2)
        x = Dense(64, kernel_regularizer=dense_regularizer, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(64, kernel_regularizer=dense_regularizer, activation='relu')(x)
        outputs = Dropout(0.1)(x)

        model = k.Model(inputs=inputs, outputs=outputs)
        return model

    def background_data_pipeline(self, input_shape):
        #Define input layer
        inputs = k.Input(shape=input_shape, name='background_data_input')
        #Define 3 Dense layers
        regularizer = l2(1e-2)
        dropout_rate = 0.2
        x = Dense(64, kernel_regularizer=regularizer, activation='relu')(inputs)
        x = Dropout(dropout_rate)(x)
        x = Dense(64, kernel_regularizer=regularizer, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(64, kernel_regularizer=regularizer, activation='relu')(x)
        outputs = Dropout(dropout_rate)(x)
        model = k.Model(inputs=inputs,outputs=outputs)
        # self.model.summary()
        return model

    def save_model_image(self,model, filename='test1.png'):
            plot_model(model, show_shapes=True, to_file=filename)

    def train_model(self, Xs, Ys, epochs, batch_size):
        #------- SPLIT DATA INTO TRAIN/TEST -------
        # Generate random state for data splitting
        rand = round(random.random()*100 + random.random()*10)
        # print('Using random state: ', rand)

        # Split data into train and test
        split = train_test_split(*(Xs + Ys), test_size=0.25, random_state=rand, shuffle=True)

        lX = len(Xs) #Get number of X input files
        lY = len(Ys) #Get number of Y output files

        X_train = [split[n*2] for n in range(lX)] #Make X_train as [np.array]
        X_test = [split[(n*2)+1] for n in range(lX)] #Make X_test as [np.array]

        if lY != 1: #If more than one Y output file
            y_train = [split[(lX*2)+(n*2)] for n in range(lY)] #Make Y_train as [np.array]
            y_test = [split[(lX*2)+(n*2)+1] for n in range(lY)] #Make Y_test as [np.array]
        else: #If only 1 Y output file
            y_train = split[-2] #Make Y_train as np.array
            y_test = split[-1] #Make Y_test as np.array
        #-------------------------------------------

        #--------------- TRAIN MODEL ---------------
        history = self.model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split=0.33)
        #-------------------------------------------

        # y_pred = self.model.predict(X_test)
        #
        #
        # y_pred_conv = (np.argmax(y_pred[0], axis=1))
        # y_true_conv = (np.argmax(y_test[0], axis=1))
        #
        # print(y_pred_conv)
        # print(y_true_conv)
        #
        # print('From test data:')
        # report = classification_report(y_true_conv, y_pred_conv)
        # print(report)

        return history, rand
