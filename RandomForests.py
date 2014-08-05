from sklearn import ensemble
from DataManipulation import X_train,y_train, y, x,X_test,y_test
import pylab as plt
from matplotlib.colors import ListedColormap
import numpy as np

def RandomForestImplementation():
    rfc = ensemble.RandomForestClassifier(n_estimators=500, compute_importances=True, oob_score=True)
    rfc.fit(X_train, y_train)
    print "Test Score Random Forests : ", rfc.score(X_train, y_train)
    print "Test Score Random Forests : ", rfc.predict(X_test)