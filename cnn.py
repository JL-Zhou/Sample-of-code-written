import keras  #library for neural network
import pandas as pd  #loading data in table form
import seaborn as sns  #visualisation
import matplotlib.pyplot as plt  #visualisation
import numpy as np  # linear algebra
from sklearn.preprocessing import normalize  #machine learning algorithm library
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
#from tensorflow.keras.layers import BatchNormalization
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.utils import np_utils

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from itertools import cycle


data = pd.read_csv("Apple_features.csv")
#print("Describing the data: ", data.describe())
#print("Info of the data:", data.info())

df = data.to_numpy()

#print(df)

np.random.shuffle(df)

X = np.delete(df, (2, 3, 4), axis=1)
y = np.delete(df, (0, 1), axis=1)
X_normalized = normalize(X)
print(X)
print(X_normalized)


#   Train test split

total_length = len(data)
train_length = int(0.8*total_length)
test_length = int(0.2*total_length)

X_train = X_normalized[:train_length]
X_test = X_normalized[train_length:]
y_train = y[:train_length]
y_test = y[train_length:]

print("Length of train set x:", X_train.shape[0], "y:",y_train.shape[0])
print("Length of test set x:", X_test.shape[0], "y:", y_test.shape[0])


########   neural networks #########

model = Sequential()
model.add(Dense(1000, input_dim=2, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=20, epochs=100, verbose=1)   ###### training


#######    prediction ############
prediction = model.predict(X_test)
length = len(prediction)
y_label = np.argmax(y_test, axis=1)
predict_label = np.argmax(prediction, axis=1)

accuracy = np.sum(y_label == predict_label)/length * 100
print("Accuracy of the dataset", accuracy)

######## confusion matrix

cm = confusion_matrix(y_label, predict_label)

sns.heatmap(cm/np.sum(cm), annot=True,
            fmt='.2%', cmap='Blues')


# sns.heatmap(cm, cmap='Greens', annot=True)
plt.title('Three class confusion Apple dataset')
plt.xlabel('True label')
plt.ylabel('Predicted label')

plt.show()



############## ROC curve

# fpr, tpr, _ = metrics.roc_curve(y_test, prediction)
n_classes = y.shape[1]

#create ROC curve
# figure()
# plt.plot(fpr, tpr)
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], prediction[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# plot roc
lw = 2
colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic to Three classes")
plt.legend(loc="lower right")
plt.show()