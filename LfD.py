#import keras #library for neural network 
import seaborn as sns #visualisation 
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd
from sklearn.preprocessing import normalize #machine learning algorithm library


#Reading data 
data=pd.read_csv("/Users/zhoujiling/Downloads/Apple_features.csv")
# print("Describing the data: ",data.describe())
# print("Info of the data:",data.info())


# print("10 first samples of the dataset:",data.head(10))
# print("10 last samples of the dataset:",data.tail(10))

#Visualisation of the dataset----------------don't know how to do 
# sns.lmplot('Score',
#            data=data,
#            fit_reg=False,
#            hue="Species",
#            scatter_kws={"marker": "D",
#                         "s": 50})
# plt.title('Score')

#convert the species into each respective category to be feed into the neural network
# print(data["Score"].unique())


data.loc[data["Score"]=="l_1","Score"]=0
data.loc[data["Score"]=="l_2","Score"]=1
data.loc[data["Score"]=="l_3","Score"]=2
# print(data.head())


data=data.iloc[np.random.permutation(len(data))]
#print(data.head())


# Converting data to numpy array in order for processing
X=data.iloc[:,1:5].values # l_1 ~ l_3 and d_score
y=data.iloc[:,0].values # question: d_score or Score
#print(y)

# print("Shape of X",X.shape)
# print("Shape of y",y.shape)
# print("Examples of X\n",X[:3])
# print("Examples of y\n",y[:3])



#Normalization
X_normalized=normalize(X,axis=0)
# print("Examples of X_normalised\n",X_normalized[:3]) . # why [:3]


#Creating train,test and validation data
total_length=len(data)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)

X_train=X_normalized[:train_length]
X_test=X_normalized[train_length:]
y_train=y[:train_length]
y_test=y[train_length:]

# print("Length of train set x:",X_train.shape[0],"y:",y_train.shape[0])
# print("Length of test set x:",X_test.shape[0],"y:",y_test.shape[0])


#Neural network module
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils

#Change the label to one hot vector
y_train=np_utils.to_categorical(y_train,num_classes=3)
y_test=np_utils.to_categorical(y_test,num_classes=3)
print("Shape of y_train",y_train.shape)
print("Shape of y_test",y_test.shape)


