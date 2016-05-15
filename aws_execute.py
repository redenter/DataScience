import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

train = load_svmlight_file("ums_svml_train.txt")
print 'loaded train data file'
train[1][np.where(train[1]==-1)] = 0

from sklearn import preprocessing, datasets, linear_model
cls = linear_model.LogisticRegression()
cls.fit(train[0], train[1])
print 'executed logistic regression'
import cPickle as pickle
pickle.dump( cls, open( "logRegModel", "wb" ))
