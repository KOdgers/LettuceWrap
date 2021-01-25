
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
from psutil import virtual_memory
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

        try:
            self.lca_type = kwargs['lca_type']
            del kwargs['lca_type']
        except:
            self.lca_type = 'Mean'

        super(LCAWrap, self).__init__(**kwargs)
        self.LCA_vals = {}
        self.last_LCA = None
        self.memory_threshold = .9


        self.Weights=False
        self.OldWeights = False


    def Fit(self,new_fitting = None,**kwargs):

        if not self.layer_names:
            self.get_layer_names()
        self.check_memory()
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
            assert self.single_epoch_size<virtual_memory()[1], "LCA will fill into swap this epoch"

            if not new_fitting:
                self.fit(**kwargs,epochs=1)
            else:
                new_fitting(**kwargs,epoch=1)
            self.get_grads()
            self.Weights = self.get_weights()

            self.last_LCA = self.get_LCA()
            self.OldWeights = self.Weights
            self.LCA_vals[j] = self.last_LCA

        # print(self.LCA_vals)



    def lca_out(self,path='',name = 'Temporary.h5'):
        temp_dict = self.LCA_vals
        temp_df = pd.DataFrame.from_dict(temp_dict)
        temp_df.to_hdf(path+name,key='df')



    def get_grads(self):
        with tf.GradientTape() as tape:
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
        return LCA

    def calculate_LCA(self):
        if not self.OldWeights:
            return 'Model hasnt been run or oldweights have been lost'
        grads = self.get_grads()
        LCA = []
        for j,jj in enumerate(self.trainable_numbers):
            lca = grads[j]*(self.Weights[jj]-self.OldWeights[jj])
            LCA.append(lca)
        return LCA

    def get_LCA(self):
        if self.lca_type=='Mean':
            return self.calculate_mean_LCA()
        elif self.lca_type=='Raw':
            return self.calculate_LCA()

    def get_layer_names(self):
        self.layer_names=[]
        for layer in self.layers:
            if layer.trainable:
                self.layer_names.append(layer.name)

    def check_memory(self,epochs=1):
        tempArray = self.get_weights()
        if self.lca_type=='Mean':
            templist = []
            for j in tempArray:
                templist.append(np.mean(j))
            size = sys.getsizeof(templist)
        elif self.lca_type =='Raw':
            size = sys.getsizeof(tempArray)
        self.single_epoch_size = size*3
        total_size = (size*epochs+3)
        available_space = virtual_memory()[1]
        if size*3 > self.memory_threshold*available_space:
            raise Exception(" Single Epoch LCA will fill memory")
        if total_size > self.memory_threshold*available_space:
            raise Exception("LCA will fill memory before completing")








1
