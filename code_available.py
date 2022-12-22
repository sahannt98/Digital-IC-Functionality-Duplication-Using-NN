#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, activations
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from numpy import asarray
from numpy import savetxt
import threading

class nnModule:
    Testmodel = tf.keras.Sequential(name='Testing Sequential Module')
    model = tf.keras.Sequential(name='Training_Sequential_Module')
    txtSavingFname="model3"
    #model 3 -> successful
    x_train=None
    x_test=None
    y_train=None
    y_test=None
    modelIn=0
    RealIn=0
    modelOut=0
    PrevOutMat=np.zeros((1,1,10))
    BreakRequestFlag=False
    TrainModelCreated=False
    value=10
    
    def __init__(self):
        print('nnModele Initiated')
        tf.random.set_seed(123)
        np.random.seed(7788)
        #tf.debugging.set_log_device_placement(True)
        #self.BreakRequest=False
        #self.value=10
    
    def NotFunc(self,x):
        if(x>0):
            return 0
        else:
            return 1

    def getDecimal(self,ArrIn):
        number=0
        for k in range(ArrIn.shape[0]):
            number= number+ (ArrIn[k]*np.power(2,k))
        return number

    def MakeBreakRequest(self):
        self.BreakRequestFlag=True
        print('Setting',self.BreakRequestFlag)

    def GenerateUART_dataset(self,SampleCount,TStepCount):
        self.BreakRequestFlag=False
        print('Generating..')
        #create input data packet structure
        StartPat = np.array([[1],[1],[0]])
        StopPat = np.array([[1]])
        DataSize=8
        NParityConfigBits=2
        DataParity=2 # 0-None,1-Odd,2-Even
        NParityBits=0
        if(DataParity!=0):
            NParityBits=1
        
        PatLength = StartPat.shape[0]+DataSize+NParityBits+StopPat.shape[0]
        n_inputs=1+NParityConfigBits
        n_outputs=DataSize+1
        #create dataset
        x_ = np.array(np.random.randint(0,high=2,size=(SampleCount,TStepCount,n_inputs)),dtype='float')
        y_= np.zeros((SampleCount,TStepCount,n_outputs),dtype='float')
        ParitySetCount=0
        for S in range(x_.shape[0]):
            if(S%100==0):
                print('.',end='')
            Idx=PatLength
            for Idx_ in range(PatLength-1,x_.shape[1]):
                if(self.BreakRequestFlag):
                    print('Exit loop')
                    return 0,0
                Q=np.array([x_[S,Idx-PatLength:Idx,0]]).T
                if(np.array_equal(StartPat,Q[0:StartPat.shape[0]])):
                    if(np.array_equal(StopPat,Q[PatLength-StopPat.shape[0]:PatLength])):
                        Data = Q[StartPat.shape[0]:PatLength-StopPat.shape[0]-NParityBits].T
                        EvenP=(np.count_nonzero(Data)%2)
                        OddP=self.NotFunc(EvenP)
                        PBrecieved=Q[PatLength-2]
                        PBexpect=0
                        DataParity=self.getDecimal(x_[S,Idx-1,1:3])
                        #set input
                        if(DataParity==1):
                            PBexpect=OddP
                        if(DataParity==2):
                            PBexpect=EvenP
                        if(DataParity==0):
                            y_[S][Idx-1][:DataSize]=Data
                            y_[S][Idx-1][DataSize]=1
                            #print("No Parity- ------------------- ", y_[S][Idx-1], PBrecieved,'-',PBexpect,'-',DataParity)
                        else:
                            if(PBrecieved==PBexpect):
                                y_[S][Idx-1][:DataSize]=Data
                                y_[S][Idx-1][DataSize]=1
                        Idx=Idx+PatLength
                    else:
                        Idx=Idx+1
                else:
                    Idx=Idx+1
                if(Idx>x_.shape[1]-1):
                    break
        print('Dataset Created')
        return x_,y_

    def AddFeedbackInput(self,x,y):
        x1=x
        y1=y
        x_ = np.zeros((x1.shape[0],x1.shape[1],x1.shape[2]+y1.shape[2]))
        x_[:,:,0:x1.shape[2]]=x1
        y_ = y1.reshape((y1.shape[0],y1.shape[1],y1.shape[2]))
        if(x.shape[1]==1):
            x_[1:x.shape[0],0,x1.shape[2]:]=y1[0:y1.shape[0]-1,0,:]
        else:
            x_[:,1:x1.shape[1],x1.shape[2]:]=y1[:,0:y1.shape[1]-1,:]
        return x_,y_

    def AddPAdding(self,x,y,xLen,yLen):
        x_=x
        y_=y
        #do padding to fix into model size
        x_padding=np.zeros((x.shape[0],x.shape[1],xLen-x.shape[2]))
        y_padding=np.zeros((y.shape[0],y.shape[1],yLen-y.shape[2]))
        x_=np.concatenate((x_,x_padding),axis=2)
        y_=np.concatenate((y_,y_padding),axis=2)
        return x_,y_

    def PrintSample(x,y):
        for k in range(100):
            for j in range(x.shape[1]):
                print(x[k][j],'\t',y[k][j])

    def GetSample(index):
        Sample_x=UART_xR[index].reshape(1,UART_xR[index].shape[0],UART_xR[index].shape[1])
        Sample_y=UART_yR[index].reshape(1,UART_yR[index].shape[0],UART_yR[index].shape[1])
        Sample_x,Sample_y=AddPAdding(Sample_x,Sample_y,modelRealInputs,self.modelOutputs)
        Sample_x,Sample_y=AddFeedbackInput(Sample_x,Sample_y)
        #print('shapes: ',Sample_x.shape,Sample_y.shape)
        return Sample_x,Sample_y

    def SaveTestModel(self):
        # serialize model to JSON
        model_json = self.Testmodel.to_json()
        with open(self.txtSavingFname+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.Testmodel.save_weights(self.txtSavingFname+".h5")
        print("Saved model to disk")

    def LoadTestModel(self):
        # load json and create model
        json_file = open(self.txtSavingFname+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        #Testmodel = model_from_json(loaded_model_json)
        # load weights into new model
        self.Testmodel.load_weights(self.txtSavingFname+".h5")
        print("Loaded model from disk")

    def LoadTrainModel(self):
        # load json and create model
        json_file = open(self.txtSavingFname+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        #Testmodel = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(self.txtSavingFname+".h5")
        print("Loaded model from disk")

    def Get3DMat3(self,I1,I2,I3):
        return np.array([[[I1,I2,I3]]])

    def CreateModel(self,n_LSTM1,Ninputs,Noutputs):
        self.modelIn=Ninputs+Noutputs
        self.modelOut=Noutputs
        self.RealIn=Ninputs
        if(self.TrainModelCreated):
            print('Previously created model')
            return self.model
        #create model
        kwargs = dict(kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.4, maxval=0.42, seed=None),bias_initializer ='uniform')
        self.model = tf.keras.Sequential(name='Training_Sequential_Module')
        self.model.add(layers.LSTM(n_LSTM1, return_sequences=True, input_shape=(None, self.modelIn),**kwargs,recurrent_initializer='Zeros' ))
        self.model.add(layers.Dense(self.modelOut,**kwargs))
        self.model.add(layers.Activation("sigmoid"))
        self.model.summary()
        self.CreateTestModel(n_LSTM1,Ninputs,Noutputs)
        print('models Created')
        self.TrainModelCreated=True
        return self.model

    def TrainModel(self,Nepoch,x,y,bs):
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size        =0.01, shuffle=False)
        print(x_train.shape,x_test.shape)
        bce = tf.keras.losses.BinaryCrossentropy()
        self.model.compile(loss=bce, optimizer='adam', metrics=['binary_accuracy'])
        es = EarlyStopping(monitor='loss', mode='min', min_delta=1e10, verbose=1,patience=10)
        history=self.model.fit(x_train, y_train, batch_size=bs, epochs=Nepoch,callbacks=[],shuffle=False,verbose=1)
        self.Testmodel.set_weights(self.model.get_weights())
        self.Testmodel.reset_states()
        self.SaveTestModel()
        scores = self.model.evaluate(x_test, y_test, verbose=2)
        print('Test loss: ',scores[0],'Test accuracy: ',scores[1])
        self.PrintTrainProgress(history)
        return history,self.Testmodel

    def PrintTrainProgress(self,history):
        from matplotlib import pyplot
        pyplot.plot(history.history['loss'], label='training loss')
        pyplot.legend()
        pyplot.show()

    def CreateTestModel(self,n_LSTM1,Ninputs,Noutputs):
        self.modelIn=Ninputs+Noutputs
        self.modelOut=Noutputs
        self.RealIn=Ninputs
        self.Testmodel = tf.keras.Sequential(
        name='Testing_Sequential_Module')
        self.Testmodel.add(layers.LSTM(n_LSTM1, return_sequences=False,stateful=True,batch_size=1, input_shape=(None, self.modelIn)))
        self.Testmodel.add(layers.Dense(self.modelOut,
        activation='sigmoid'))
        self.Testmodel.summary()
        self.Testmodel.reset_states()
        print('Testing model Created.')

    def SampleTest(self,x,y):
        for S in range(10):
            for Idx in range(x.shape[1]):
                X = np.array([[x[S][Idx]]])
                predicted_stateful = (Testmodel.predict(X).flatten()>0.5)*1
                out=y[S][Idx]
                print( np.array(X.reshape(X.shape[2]),dtype='int'),predicted_stateful,np.array(out,dtype='int'))

    def SampleTestIndividual():
        PrevOutMat=np.zeros((1,1,10))
        self.Testmodel.reset_states()
        for S in range(10):
            for Idx in range(UART_x.shape[1]):
                In = np.array([[x[S][Idx]]])[:,:,:3]
                Result = GetModelResult(Testmodel,In)
                out=UART_y[S][Idx]
                print( np.array(In.reshape(3),dtype='int'),' ',Result,' ',out)

    def GetModelResult(self,InputMat):
        global PrevOutMat
        InputMatPadded,dummy=self.AddPAdding(InputMat,InputMat,self.RealIn,self.modelOut)
        InputMatFB= np.concatenate((InputMatPadded,self.PrevOutMat),axis=2)
        modelResult = (self.Testmodel.predict(InputMatFB).flatten()>0.5)*1
        self.PrevOutMat = modelResult.reshape(1,1,modelResult.shape[0])
        return modelResult

    def SaveSampleDataSet(self,x,y,name):
        # define data
        np.save(name+'_x',x)
        np.save(name+'_y',y)
        print('Saving..')

    def LoadSampleDataSet(self,name):
        x=np.load(name+'_x.npy')
        y=np.load(name+'_y.npy')
        print(name+' file opened')
        print(x.shape,y.shape)
        return x,y

    def LoadSampleDataSetFromPath(self,strPath):
        from pathlib import Path
        fpath=Path(strPath)
        baseName=str(fpath.name).split('_',1)[0]
        x=np.load(baseName+'_x.npy')
        y=np.load(baseName+'_y.npy')
        print(baseName +' file opened')
        return x,y

    def SaveSampleDataSetFromParh(self,x,y,strPath):
        from pathlib import Path
        fpath=Path(strPath)
        baseName=str(fpath.name).split('_',1)[0]
        # define data
        np.save(baseName+'_x',x)
        np.save(baseName+'_y',y)
        print(baseName + ' Saving..')

    def AppendSampleDataSetFromPath(self,x,y,strPath):
        xNew,yNew=self.LoadSampleDataSetFromPath(strPath)
        xAdded=np.concatenate((x,xNew),axis=0)
        yAdded=np.concatenate((y,yNew),axis=0)
        print('new: ',xNew.shape,yNew.shape)
        print('add: ',xAdded.shape,yAdded.shape)
        print('in:',x.shape,y.shape)
        return xAdded,yAdded
