print(__doc__)

import numpy as np
import pylab as pythonLab
from sklearn import svm

twitterData = np.genfromtxt('/Users/Kaushy/Courses/Project/SupportVectorMachine/MyDataSet.csv', delimiter=',')

#Dropping the First Header in the Row
twitterData = twitterData[1:]
NumberOfRowsOfData, NumberOfColumnsOfFeatures = twitterData.shape

# the target classification is the First column
TargetClassificationRow = 1
NumberOfColumnsOfFeatures = NumberOfColumnsOfFeatures-TargetClassificationRow

# Test Using only the first two features (Slicing in python 0,1,2,3,4,5 etc when you say 1:3 that is 1 and 2 in this case. So not 0 and not 3)
XValue = twitterData[:,1:5]
YValueColumnVector = twitterData[:,:TargetClassificationRow]

# Create a 1D array off the column vector we get from YValue
YValue = YValueColumnVector.ravel()

# Fit model to this and get the hyperplane. SVC is Support Vector Classification. Can also chose kernla to be poly, rbf, sigmoid, precomputed or callable depending on the shape of the
myClassifier = svm.SVC(kernel='linear')
myClassifier.fit(XValue, YValue)

#Getting the seperating Hyper Planes
avgDifferenceBetweenTwoPoints = myClassifier.coef_[0]
# Slope or Gradient m = d(y) / d(x)
a = -avgDifferenceBetweenTwoPoints[1] / avgDifferenceBetweenTwoPoints[0]
xPlane = np.linspace(-40, 100)
yPlane = a * xPlane - myClassifier.intercept_[0] / avgDifferenceBetweenTwoPoints[0] # y = mx + c

#The following commands bring up the cooridnates of the two planes.
print xPlane
print yPlane

# get the separating hyperplane using weighted classes
weightedmyClassifier = svm.SVC(kernel='linear', class_weight={0: 20})
weightedmyClassifier.fit(XValue, YValue)

weightedavgDifferenceBetweenTwoPoints = weightedmyClassifier.coef_[0]
weightedA = -weightedavgDifferenceBetweenTwoPoints[1] / weightedavgDifferenceBetweenTwoPoints[0]
weightedyPlane = weightedA * xPlane - weightedmyClassifier.intercept_[0] / weightedavgDifferenceBetweenTwoPoints[1]

# plot separating hyperplanes
hyperPlaneWithNoWeights = pythonLab.plot(xPlane, yPlane, '-r', label='no weights')
hyperPlaneWithWeights = pythonLab.plot(xPlane, weightedyPlane, '--b', label='with weights')

# here c is color and cmap is color map. -g is solid green line. --b is double dash blue line and we are getting the first and second dimensions of the XValue variable which we defined
# at the top. So XValue 0 would be technically the second column and XValue 1 would be the third but anws
pythonLab.scatter(XValue[:, 0], XValue[:, 1], c=YValue, cmap=pythonLab.cm.Paired)
pythonLab.legend()

pythonLab.axis('tight')
pythonLab.show()
