import numpy as np
import pandas as pd
import os
import seaborn as sns
from scipy.stats import skew
from sklearn import cross_validation

os.getcwd()
os.chdir("E:\Python_Dataset\Leaf Classification")

train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")

train.describe()

train.dtypes

from sklearn.preprocessing import LabelEncoder

def encode(train,test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)
    classes = list(le.classes_)
    test_id = test.id

    train=train.drop(['id','species'],axis=1)
    test=test.drop(["id"],axis=1)
    return train,test,labels,test_id,classes

train,test,labels,test_id,classes = encode(train,test)

from sklearn.cross_validation import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
from sklearn.neighbors import KNeighborsClassifier 

N = [1,2,3,4,5,6,7,8,9,10]
kscores = []

for k in N:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_validation.cross_val_score(knn,X_train, y_train,scoring="accuracy")
    kscores.append(scores.mean())    
    
print (kscores)

kn=KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)

pred=kn.predict_proba(test)

# Format DataFrame
submission = pd.DataFrame(pred, columns=classes)
submission.insert(0, 'id', test_id)
submission.reset_index()

# Export Submission
submission.to_csv('submission1.csv', index = False)

########random forest############
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf = rf.fit(X_train,y_train)
predic = rf.predict_proba(test)

# Format DataFrame
submission = pd.DataFrame(predic, columns=classes)
submission.insert(0, 'id', test_id)
submission.reset_index()

# Export Submission
submission.to_csv('submission1.csv', index = False)


    



