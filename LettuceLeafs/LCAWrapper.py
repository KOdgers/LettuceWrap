
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow.keras.optimizers as opts
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.utils import shuffle



from tensorflow.python.util import tf_decorator

from sklearn.model_selection import train_test_split

class LCAWrap(Model):

    def __init__(self,*args,**kwargs):

        try:
            self.layer_names = kwargs['layer_names']
            del kwargs['layer_names']
        except:
            self.layer_names = None

        super(LCAWrap, self).__init__(**kwargs)
        self.LCA_vals = {}
        self.last_LCA = None


        self.Weights=False
        self.OldWeights = False


    def Fit(self,new_fitting = None,**kwargs):
        if not self.layer_names:
            self.get_layer_names()
        epochs = kwargs['epochs']
        del kwargs['epochs']

        self.OldWeights = self.get_weights()
        if not kwargs['validation_data']:
            X_train,Y_train,x_test,y_test = train_test_split(kwargs['x'],kwargs['y'],kwargs['validation_split'])
            kwargs['x']=X_train
            kwargs['y']=Y_train
            kwargs['validation_data'] = (x_test,y_test)

            del kwargs['validation_split']
        else:
            x_test= kwargs['validation_data'][0]
            y_test = kwargs['validation_data'][1]
        self.x_test = x_test
        self.y_test = y_test
        # print('Y_test shape:',y_test.shape)

        for j in range(0,epochs):
            if not new_fitting:
                self.fit(**kwargs,epochs=1)
            else:
                new_fitting(**kwargs,epoch=1)
            self.get_grads()
            self.Weights = self.get_weights()
            self.last_LCA = self.calculate_mean_LCA()
            self.OldWeights = self.Weights
            self.LCA_vals[j] = self.last_LCA

        # print(self.LCA_vals)



    def lca_out(self,path='',name = 'Temporary.h5'):
        temp_dict = self.LCA_vals
        temp_df = pd.DataFrame.from_dict(temp_dict)
        temp_df.to_hdf(path+name,key='df')

    # def run_epoch(self):
    #
    #     numUpdates = int(self.XTrain.shape[0] / self.BatchSize)
    #     def step(X, y):
    #         with tf.GradientTape() as tape:
    #             pred = self.Model(X)
    #             loss = categorical_crossentropy(y, pred)
    #         grads = tape.gradient(loss, self.Model.trainable_variables)
    #         self.opt.apply_gradients(zip(grads, self.Model.trainable_variables))
    #
    #     for i in range(0, numUpdates):
    #         start_index = i * self.BatchSize
    #         end_index = start_index + self.BatchSize
    #         step(self.XTrain[start_index:end_index], self.YTrain[start_index:end_index])
    #     pred = self.Model(self.xtest)
    #     acc = categorical_accuracy(self.ytest,pred)
    #     print(np.mean(acc))
    #     self.loss = np.mean(acc)
    #     #
    #     self.XTrain,self.YTrain = shuffle(self.XTrain,self.YTrain)

    def lca_epoch(self,run_epoch=None):

        # compute the number of batch updates per epoch
        if run_epoch:
            run_epoch(self.Model,self.opt,self.XTrain,self.YTrain)
        else:
            self.run_epoch()
        self.Weights = self.get_weights()
        self.last_LCA = self.calculate_LCA()
        self.OldWeights = self.Weights
        return self.last_LCA

    def get_grads(self):
        with tf.GradientTape() as tape:
            # tape.watch(self.trainable_variables)
            pred = self(self.x_test)
            loss = categorical_crossentropy(self.y_test,pred)
        return tape.gradient(loss,self.trainable_variables)


    def get_weights(self):
        listOfVariableTensors = self.trainable_weights
        Weights = [[]]
        self.trainable_numbers = []
        print([listOfVariableTensors[i].name for i in range(0,len(listOfVariableTensors))])

        for l in range(0, len(listOfVariableTensors)):
            if listOfVariableTensors[l].name.split("/")[0] in self.layer_names:
                self.trainable_numbers.append(l)
                Weights.append(listOfVariableTensors[l].value())
        del Weights[0]
        return Weights

    def calculate_mean_LCA(self):
        if not self.OldWeights:
            return 'Model hasnt been run or oldweights have been lost'
        grads = self.get_grads()
        LCA = []
        for j,jj in enumerate(self.trainable_numbers):
            lca = grads[j]*(self.Weights[jj]-self.OldWeights[jj])
            LCA.append(np.mean(lca))
            # print(LCA)
        return LCA

    def get_layer_names(self):
        self.layer_names=[]
        for layer in self.layers:
            if layer.trainable:
                self.layer_names.append(layer.name)




1
