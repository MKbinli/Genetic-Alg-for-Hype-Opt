from __future__ import absolute_import, division, print_function, unicode_literals

import random
import pandas as pd
import numpy as np
import copy
import gc
import sys
import multiprocessing as mp
import itertools
from numba import jit
import tensorflow as tf
import matplotlib.pyplot as plt
from openpyxl.workbook import Workbook
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
import random
import os
if sys.version_info[0] >= 3:
    import PySimpleGUI as sg
else:
    import PySimpleGUI27 as sg

tf.compat.v1.enable_eager_execution()


class GeneticBuilder():
    chromozomeKeys=[]
    stringKeys=[]
    chromozomeValues=[]
    pureGeneChromoSize=None
    train_stats=None
    labelName="target"
    discardList=["Timestamp","Date"]

    def __init__(self):
        self.setFeatures()
        dataset_path='xxx.csv'

        df1 = pd.read_csv(dataset_path)
        dataset = df1.copy()
        dataset=dataset.dropna()
        dataset=dataset.drop(columns=self.discardList)
        train_dataset = dataset.copy()
        sizeOfData=len(dataset)
        percentageDataSize=int((sizeOfData*80)/100)
        partedData=dataset[:percentageDataSize].copy()

        self.train_stats=train_dataset.describe()
        self.train_stats = self.train_stats.transpose()
        normed_train_data = self.norm(train_dataset)
        train_labels = normed_train_data.pop(self.labelName)
        predictDayPeriods=10
        ##train_labels = self.labelEncodingMultiStep(train_labels, nOut=predictDayPeriods, dropnan=True)
        train_labels =self.labelEncoding(train_labels)

        #Change the data structure to Numpy array
        normed_train_data = np.array(normed_train_data)
        train_labels = np.array(train_labels)#
        #Get the feature size before concatenate them
        self.featureSize=normed_train_data.shape[1]
        #Last 20 days will be hold out data
##        normed_train_data=normed_train_data[:-(predictDayPeriods-1)]
        #Concatenate train data and labels all together as an array
        self.xy_train=np.concatenate([normed_train_data,train_labels],axis=1)
    def rangeGenerator(self,low,high,freq):
        arrayOfVal=np.arange(low,high,freq)
        return arrayOfVal
    

    def norm(self,x):#NORMALIZE THE TRAINING DATA
        return (x-self.train_stats['mean'])/self.train_stats['std']

    def normSecond(self,x):
        return (x-self.train_stats['min'])/(self.train_stats['max']-self.train_stats['min'])

    def normAdverse(self,x):#Column names must be same
        return (x*self.train_stats['std'])+self.train_stats['mean']

    def normSecondAdverse(self,x):#Column names must be same
        return (x*(self.train_stats['max']-self.train_stats['min']))+self.train_stats['min']

    def labelEncoding(self,data_labels):#label encoding
        labels = np.empty((0, 3))
        for i in data_labels:
            if i == 0:
                labels = np.append(labels, [[1,0,0]], axis=0)
            elif i==1:
                labels = np.append(labels, [[0,1,0]], axis=0)
            elif i==2:
                labels = np.append(labels, [[0,0,1]], axis=0)
        return labels

    def labelEncodingUniStep(self,data_labels):#label encoding
        labels = np.empty((0, 1))
        for i in data_labels:
            labels = np.append(labels, [[i]], axis=0)

        return labels

    def labelEncodingMultiStep(self,data, nOut=1, dropnan=True):
            nVars = 1 
            df = pd.DataFrame(data)
            cols, names = list(), list()
            
            for i in range(0, nOut):
                    cols.append(df.shift(-i))
                    if i == 0:
                            names += [('var%d(d)' % (j+1)) for j in range(nVars)]
                    else:
                            names += [('var%d(d+%d)' % (j+1, i)) for j in range(nVars)]
            
            pred = pd.concat(cols, axis=1)
            pred.columns = names

            if dropnan:
                    pred.dropna(inplace=True)
            pred=np.array(pred)
            return pred
    
    def splitSequences(self,sequences, nSteps, featureSize):
        X = list()
        y = list()
        for i in range(len(sequences)):
                endXi = i + nSteps
                if endXi > len(sequences):
                        break
                seqX, seqY = sequences[i:endXi, :featureSize], sequences[endXi-1, featureSize:]
                X.append(seqX)
                y.append(seqY)
        return np.array(X), np.array(y)

    def windowDataset(self,series, winSize, batchSize=32,
                   shuffBuffer=1000):
        dataset = tf.data.Dataset.from_tensor_slices(series)
        
        dataset = dataset.window(winSize + 1)
        dataset = dataset.flat_map(lambda window: window.batch(winSize + 1))
        #dataset = dataset.shuffle(shuffBuffer)
        dataset = dataset.map(lambda window: (window[:-3], window[-3]))
        dataset = dataset.batch(batchSize).prefetch(1)
        return dataset
    def setFeatures(self):
        temp=self.rangeGenerator(0.1,1,0.1)
        momentBeta=[]
        for val in temp: momentBeta.append(float(val))
        optVal=[0,1,2]
        batchVal=[0,1]
        hidLayerSize=[0,1,2]
        temp=self.rangeGenerator(2,101,1)
        windowSize=[]
        for val in temp: windowSize.append(int(val))
        dropVal=[0,0.1,0.2,0.3,0.4]
        regVal=np.logspace(-8, -1, 20)
        lRate=np.logspace(-8, -1, 20)
        initilizerWeight=['he_uniform','orthogonal','truncated_normal','identity','lecun_uniform','lecun_normal','glorot_uniform','random_uniform']
        activationFunction=[tf.nn.tanh,tf.nn.relu,tf.nn.leaky_relu,tf.nn.elu, tf.nn.selu,tf.nn.relu6]#
        temp=self.rangeGenerator(5,220,5)
        neuronSize=[]
        for val in temp: neuronSize.append(int(val))
        dictOfParams={"momentBeta":momentBeta,"optVal":optVal,
                      "batchVal":batchVal,"hidLayerSize":hidLayerSize,
                      "windowSize":windowSize,"dropVal":dropVal,
                      "regVal":regVal,"lRate":lRate,
                      "initilizerWeight":initilizerWeight,
                      "activationFunction":activationFunction,
                      "neuronSize":neuronSize}

        for key in dictOfParams:
            self.chromozomeKeys.append(key)
            self.stringKeys.append(str(key))
        for fea in dictOfParams.values():
            self.chromozomeValues.append(fea)

        self.pureGeneChromoSize=len(self.chromozomeValues)



    def getRandomIndex(self,sizeChromozomeValuesList):#Takes the size of List of Chromo Vals
        return int(random.uniform(0,sizeChromozomeValuesList))

    def randomPointGenerator(self):
        geneNo=self.pureGeneChromoSize-1#discard the last           point of Chromo for crossing over point
        return int(random.uniform(1,geneNo))#             and first

    def randomGeneSelector(self):
        geneNo=self.pureGeneChromoSize
        return int(random.uniform(0,geneNo))#include first and last gene for mutation
    
        
    def randomMutator(self):#Probability Supplier
        return int(random.uniform(1,11))

    def geneDetector(self):
        randomGene=self.randomGeneSelector()
        singleRanChromoVal=self.chromozomeValues[randomGene]
        gene=self.getRandomIndex(len(singleRanChromoVal))
        return gene,randomGene


    def createHiddenLayerUnits(self,neuronSize=100, model=keras.Sequential(), regVal=0.0, dropVal=0.0,hidLayerSize=0, batchNorm=0,initilizerWeight='lecun_uniform',activationFunction=tf.nn.tanh):
        if(batchNorm==0):
            for i in range(hidLayerSize):
                model.add(layers.LSTM(neuronSize, return_sequences=True, kernel_initializer=initilizerWeight, kernel_regularizer=keras.regularizers.l1(regVal), activation=activationFunction))
                model.add(keras.layers.Dropout(dropVal))
        elif(batchNorm==1):
            for i in range(hidLayerSize):
                model.add(layers.LSTM(neuronSize, return_sequences=True, kernel_initializer=initilizerWeight, kernel_regularizer=keras.regularizers.l1(regVal), activation=activationFunction))
                model.add(layers.BatchNormalization())
                model.add(keras.layers.Dropout(dropVal))
        return model

    def populationGeneration(self):#100 population for begining
        initialPopulation=[]
        amountofPop=100
        for i in range(amountofPop):
            initialPopulation.append(np.zeros([self.pureGeneChromoSize]))
            counter=0
            for feaList in self.chromozomeValues:
                genVal=self.getRandomIndex(len(feaList))
                initialPopulation[i][counter]=int(genVal)
                counter=counter+1
        iniPop=np.array(initialPopulation)
        return iniPop

    def fitnessCalculationLoop(self,stKey,chVal,xy_train,featureSize,*arg):
        initialPopulation=[*arg]
        initialPopulation=initialPopulation[0]
        featureDict=dict()
        for indFea in range(len(initialPopulation)):
            if indFea==0:
                featureDict={stKey[indFea]:chVal[indFea][int(initialPopulation[indFea])]}
            else:
                featureDict={**featureDict,stKey[indFea]:chVal[indFea][int(initialPopulation[indFea])]}

            
        time = np.arange(len(xy_train))
        split_time = int(len(xy_train) * 0.80)
        time_train = time[:split_time]
        time_valid = time[split_time-featureDict["windowSize"]:]
        x_valid = xy_train[split_time-featureDict["windowSize"]:]
        x_train = xy_train[:split_time]
        #Data preparation for LSTM.It produces train and test data in the shape of LSTM input shape and their labels for the output shape.
        trainSplit,trlabels= self.splitSequences(x_train, featureDict["windowSize"], featureSize)
        testSplit,tslabels= self.splitSequences(x_valid, featureDict["windowSize"], featureSize)
        #output size

        outputSize=3

        print('SELECTED CROMOZOME : '+str(initialPopulation))
        print('MOMENTBETA VALUE : '+str(featureDict["momentBeta"]))
        print('OPT VALUE : '+ str(featureDict["optVal"]))
        print('BATCH VALUE (TRUE OR FALSE) : '+str(featureDict["batchVal"]))
        print('HIDDEN LAYER : '+str(featureDict["hidLayerSize"]))
        print('DROPOUT VALUE : '+str(featureDict["dropVal"]))
        print('WINDOW VALUE : '+str(featureDict["windowSize"]))
        print('REGULAR VALUE : '+ str(featureDict["regVal"]))
        print('LEARNING RATE : '+str(featureDict["lRate"]))
        print('WEIGHT INIT : '+ featureDict["initilizerWeight"])
        print('ACT FUNC. : '+ str(featureDict["activationFunction"]))
        print('NEURON SIZE : '+ str(featureDict["neuronSize"]))
        model=keras.Sequential()
        if (featureDict["hidLayerSize"]!=0):
            model.add(layers.LSTM(featureDict["neuronSize"], return_sequences=True, kernel_initializer=featureDict["initilizerWeight"], kernel_regularizer=keras.regularizers.l1(featureDict["regVal"]), activation=featureDict["activationFunction"], input_shape=[featureDict["windowSize"],featureSize]))
        if(featureDict["hidLayerSize"]==0):
           if(featureDict["batchVal"]==0):
                model.add(layers.LSTM(featureDict["neuronSize"], kernel_initializer=featureDict["initilizerWeight"], kernel_regularizer=keras.regularizers.l1(featureDict["regVal"]), activation=featureDict["activationFunction"], input_shape=[featureDict["windowSize"],featureSize]))
                model.add(layers.Dropout(featureDict["dropVal"]))
           else:
                model.add(layers.LSTM(featureDict["neuronSize"], kernel_initializer=featureDict["initilizerWeight"], kernel_regularizer=keras.regularizers.l1(featureDict["regVal"]), activation=featureDict["activationFunction"], input_shape=[featureDict["windowSize"],featureSize]))
                model.add(layers.BatchNormalization())
                model.add(layers.Dropout(featureDict["dropVal"]))
        elif featureDict["hidLayerSize"]>=1:
            for i in range(featureDict["hidLayerSize"]):
                if(featureDict["batchVal"]==0):
                    model.add(layers.Dropout(featureDict["dropVal"]))
                    model=createHiddenLayerUnits(neuronSize=featureDict["neuronSize"],model=model, regVal=featureDict["regVal"], dropVal=featureDict["dropVal"],hidLayerSize=featureDict["hidLayerSize"],batchNorm=featureDict["batchVal"], initilizerWeight=featureDict["initilizerWeight"],activationFunction=featureDict["activationFunction"])
                elif(featureDict["batchVal"]==1):
                    model.add(layers.BatchNormalization())
                    model.add(layers.Dropout(featureDict["dropVal"]))
                    model=self.createHiddenLayerUnits(neuronSize=featureDict["neuronSize"],
                                                      model=model, regVal=featureDict["regVal"],
                                                      dropVal=featureDict["dropVal"],
                                                      hidLayerSize=featureDict["hidLayerSize"],
                                                      batchNorm=featureDict["batchVal"],
                                                      initilizerWeight=featureDict["initilizerWeight"],
                                                      activationFunction=featureDict["activationFunction"])
            model.add(layers.LSTM(featureDict["neuronSize"], kernel_initializer=featureDict["initilizerWeight"], kernel_regularizer=keras.regularizers.l1(featureDict["regVal"]), activation=featureDict["activationFunction"]))
            if(featureDict["batchVal"]==0):
                model.add(layers.Dropout(featureDict["dropVal"]))
                                         
            elif(featureDict["batchVal"]==1):
                model.add(layers.BatchNormalization())
                model.add(layers.Dropout(featureDict["dropVal"]))
        
        model.add(layers.Dense(outputSize, activation=tf.nn.softmax))

        if (featureDict["optVal"]==0):
            print('ADAM OPTIMIZER')
            optimizer = tf.keras.optimizers.Adam(learning_rate=featureDict["lRate"], beta_1=featureDict["momentBeta"], beta_2=0.999)
        elif (featureDict["optVal"]==1):
            print('SGD OPTIMIZER')
            optimizer = tf.keras.optimizers.SGD(learning_rate=featureDict["lRate"], momentum=featureDict["momentBeta"])
            
        elif (featureDict["optVal"]==2):
            print('RMSPROP OPTIMIZER')
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=featureDict["lRate"], momentum=featureDict["momentBeta"])

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])


        print('Training has begun!!!')
        stop_noimprovement = keras.callbacks.EarlyStopping(patience=4)
        history=model.fit(trainSplit,trlabels, epochs=10,
                          validation_data=(testSplit,tslabels),verbose=2,
                          callbacks=[stop_noimprovement])
        sizeofvalmae=len(history.history['val_acc'])-1#['val_loss']
        sizeoftrmae=len(history.history['acc'])-1#['loss']
        valmae=history.history['val_acc'][sizeofvalmae]#returns last val_loss
        trmae=history.history['acc'][sizeoftrmae]#same manner above
        allResults=pd.DataFrame(columns=list(range(len(initialPopulation))),index=[0])
        for i in range(len(initialPopulation)):
            allResults[i].iat[0]=initialPopulation[i]
        allResults['val_loss']=valmae
        allResults['loss']=trmae

        del model
        keras.backend.clear_session()
            
        gc.collect()            
        
        for key in featureDict:
            allResults[key]=featureDict[key]
            
        allResults=pd.concat([initialPopulation,allResults],axis=1)
        return allResults

    def fitnessCalculation(self,initialPopulation):

        allResults=None

        pool = mp.Pool(processes=mp.cpu_count())
        new_results = pool.starmap(self.fitnessCalculationLoop,
                                   zip(itertools.repeat(self.stringKeys),
                                       itertools.repeat(self.chromozomeValues),
                                       itertools.repeat(self.xy_train),
                                       itertools.repeat(self.featureSize),
                                       initialPopulation))


        if allResults==None:
            allResults=pd.DataFrame(columns=list(new_results[0].columns),dtype=object)
        for nwR in new_results:
            allResults=pd.concat([allResults,nwR],axis=0)
            


        sortParams=['val_loss','loss']
        sortParamsAscenStat=[False,False]
        allResults.index=range(0,allResults.shape[0])
        bestChromozome=allResults.sort_values(by=sortParams,ascending=sortParamsAscenStat)
        calculatedPopulation=bestChromozome.copy()
        bestChromo=bestChromozome[0:1]
   
        return calculatedPopulation,bestChromo
        
    def tournamentSelection(self,calculatedPopulation):

        selectionParentSize=30
        amountofSelection=3
        parametersCount=self.pureGeneChromoSize
        paramColumnsName=list(range(parametersCount))
        selectedParentPopulation=np.empty([selectionParentSize,parametersCount])
        for i in range (selectionParentSize):
            tempDf=pd.DataFrame(columns=list(calculatedPopulation.columns),dtype=object)
            for j in range(amountofSelection):
               k=int(random.uniform(0,len(calculatedPopulation)))
               tempDf=pd.concat([tempDf,calculatedPopulation.iloc[k:k+1]],axis=0)

            sortParams=['val_loss','loss']
            sortParamsAscenStat=[False,False]
            tempDf=tempDf.sort_values(by=sortParams,ascending=sortParamsAscenStat)
            fivePop=np.array(tempDf[paramColumnsName])
            selectedParentPopulation[i]=fivePop[0]
        sortedPopulation=selectedParentPopulation[:,paramColumnsName]
        return sortedPopulation

    def crossingOver(self,sortedPopulation):

        chromozomeSize=self.pureGeneChromoSize-1
        selectionParentSize=4
        shapeRows=sortedPopulation.shape[1]
        backwardMove=len(sortedPopulation)
        topPop=np.zeros([selectionParentSize,shapeRows])
        for i in range (selectionParentSize):
            k=int(random.uniform(0,len(sortedPopulation)))
            topPop[i]=sortedPopulation[k]
        m=0
        while(m<16):
            crossplitPoint=self.randomPointGenerator()
            tempA=sortedPopulation[m,:crossplitPoint]
            tempB=sortedPopulation[m+1,:crossplitPoint]
            sortedPopulation[m,:crossplitPoint]=tempB
            sortedPopulation[m+1,:crossplitPoint]=tempA
            topPop=np.concatenate([topPop,sortedPopulation[m:m+1,:]],axis=0)
            topPop=np.concatenate([topPop,sortedPopulation[m+1:m+2,:]],axis=0)
            crossplitPoint=self.randomPointGenerator()
            tempA=sortedPopulation[m,:crossplitPoint]
            tempB=sortedPopulation[-(m+1),:crossplitPoint]
            sortedPopulation[m,:crossplitPoint]=tempB
            sortedPopulation[-(m+1),:crossplitPoint]=tempA
            topPop=np.concatenate([topPop,sortedPopulation[m:m+1,:]],axis=0)
            topPop=np.concatenate([topPop,sortedPopulation[(backwardMove-1-m):(backwardMove-m),:]],axis=0)
            crossplitPoint=self.randomPointGenerator()
            tempA=sortedPopulation[-(m+1),:crossplitPoint]
            tempB=sortedPopulation[-(m+2),:crossplitPoint]
            sortedPopulation[-(m+1),:crossplitPoint]=tempB
            sortedPopulation[-(m+2),:crossplitPoint]=tempA
            topPop=np.concatenate([topPop,sortedPopulation[(backwardMove-1-m):(backwardMove-m),:]],axis=0)
            topPop=np.concatenate([topPop,sortedPopulation[(backwardMove-2-m):(backwardMove-1-m),:]],axis=0)
            m=m+1

        childPopulation=topPop
        return childPopulation


    def mutatingPopulation(self,childPopulation):#mutation is %20

        sizeofPop=len(childPopulation)
        for i in range(sizeofPop):
            probabilityMutate=self.randomMutator()
            if probabilityMutate==1 or probabilityMutate==5 :
                geneValue,randomGene=self.geneDetector()
                childPopulation[i,randomGene]=geneValue
            else:
                continue

        return childPopulation


if __name__ == '__main__':
    GAoptimizer=GeneticBuilder()
    initialPopulation=GAoptimizer.populationGeneration()
    iterationSize=5
    for i in range(iterationSize):
        #np.random.seed(42)
        calculatedPopulation,tempBestChromo=GAoptimizer.fitnessCalculation(initialPopulation)
        del initialPopulation
        sortedPopulation=GAoptimizer.tournamentSelection(calculatedPopulation)
        del calculatedPopulation
        childPopulation=GAoptimizer.crossingOver(sortedPopulation)
        del sortedPopulation
        initialPopulation=GAoptimizer.mutatingPopulation(childPopulation)
        del childPopulation
        if i==0:
            allResults=pd.DataFrame(columns=list(tempBestChromo.columns), dtype=object)
        allResults=pd.concat([allResults,tempBestChromo],axis=0)
    sortParams=['val_loss','loss']
    sortParamsAscenStat=[False,False]
    allResults=allResults.sort_values(by=sortParams,ascending=sortParamsAscenStat)
    allResults.to_excel("gahyperopresults.xlsx",index=False)
    






        
