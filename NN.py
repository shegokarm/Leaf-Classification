#Importing libraries
import os
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#Changing directory
os.getcwd()
os.chdir("E:\Python_Dataset\Leaf Classification")

#Importing data
train = pd.read_csv("train.csv")

parent_data = train.copy()

test = pd.read_csv("test.csv")

#y = train['species']
y= train.pop('species')
#Converting categorical values to numerical values
y = LabelEncoder().fit(y).transform(y)

y_cat = to_categorical(y)

#id = train['id']
id = train.pop('id')

X = StandardScaler().fit(train).transform(train)

model = Sequential()
model.add(Dense(256,input_dim=192))
model.add(Activation('relu'))
model.add(Dense(99))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

history = model.fit(X,y_cat,batch_size=64,nb_epoch=10,verbose=0)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'],'o-')
plt.xlabel('Number of Iterations')
plt.ylabel('Categorical Crossentropy')
plt.title('Train Error vs Number of Iterations')

test_id = test.pop('id')

test = StandardScaler().fit(test).transform(test)

yPred = model.predict_proba(test)
    
## Converting the test predictions in a dataframe as depicted by sample submission
yPred = pd.DataFrame(yPred,index=test_id,columns=sorted(parent_data.species.unique()))

fp = open('submission.csv','w')

fp.write(yPred.to_csv())

# Export Submission
yPred.to_csv('submission.csv', index = True)






