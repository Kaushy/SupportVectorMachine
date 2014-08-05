print(__doc__)

from SVM import SupportVectorMachineImplementation
from RandomForests import RandomForestImplementation
from sklearn.linear_model import LogisticRegression
from DataManipulation import X_train,y_train, y, x,X_test,y_test,X,y
import matplotlib.pyplot as graphPlotter
from utility import plot_decision_boundary

#-------------------------------------------------------------------------------------------- LR -----------------------------------------------------#
## ------------ Data Pruning ----------------##
# Import logistic regression from scikit-learn and generate a classification object.
logreg = LogisticRegression(class_weight='auto')

#Now let's fit the logistic regression model to the training data, to see if we can do it automatically:
logreg.fit(X_train, y_train)

print "Result Parameters : ", logreg.get_params(deep='True')
#print "Confidence Scores : ", logreg.decision_function(X_train)
#
# First, we have a look at how logistic regression did on the training set.
#We do this by now setting colors using the predicted class.

y_pred_train = logreg.predict(X_train)
graphPlotter.scatter(X_train[:, 0], X_train[:, 1],X_train[:, 2], c=y_pred_train)
plot_decision_boundary(logreg, X)
#graphPlotter.show()

print "Accuracy on training set LR:", logreg.score(X_train, y_train)

y_pred_test = logreg.predict(X_test)
graphPlotter.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, marker='^')
graphPlotter.scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train)
plot_decision_boundary(logreg, X)
#graphPlotter.show()

#print "Accuracy on test set LR:", logreg.score(X_test, y_test)
print "Prediction : ", logreg.predict(X_test)

SupportVectorMachineImplementation()
RandomForestImplementation()

################################################################################# SVM ########################################################
#n_sample = len(X)
#
#numpy.random.seed(0)
#order = numpy.random.permutation(n_sample)
#
## Arranging X and Y in the order from above
#X = X[order]
#y = y[order].astype(numpy.float)
#
## do a cross validation
#nighty_precent_of_sample = int(.9 * n_sample)
#X_train = X[:nighty_precent_of_sample]
#y_train = y[:nighty_precent_of_sample]
#X_test = X[nighty_precent_of_sample:]
#y_test = y[nighty_precent_of_sample:]
#
#
## create a list of the types of kerneks we will use for your analysis
#types_of_kernels = ['linear', 'rbf', 'poly']
#
## specify our color map for plotting the results
#color_map = graphPlotter.cm.RdBu_r
#
## fit the model
#for fig_num, kernel in enumerate(types_of_kernels):
#    classifier = svm.SVC(kernel=kernel, gamma=3)
#    classifier.fit(X_train, y_train)
#    
#    graphPlotter.figure(fig_num)
#
#    graphPlotter.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=color_map)
#
#    # circle out the test data
#    graphPlotter.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
#
#    graphPlotter.axis('tight')
#    x_min = X[:, 0].min()
#    x_max = X[:, 0].max()
#    y_min = X[:, 1].min()
#    y_max = X[:, 1].max()
#    p_min = X[:, 2].min()
#    p_max = X[:, 2].max()
#
#    XX, YY, PP = numpy.mgrid[x_min:x_max:200j, y_min:y_max:200j, p_min:p_max:200j] # Look at screen shot. 200j in first instance will be number of rows second 200j is columns.
#    #print numpy.c_[XX.ravel(), YY.ravel(), PP.ravel()]
#    print "------------------------------------------------------------------"
#    Z = classifier.decision_function(numpy.c_[XX.ravel(), YY.ravel(), PP.ravel()])
#
#    # put the result into a color plot
#    print XX.shape
#    print PP.shape
#    Z = Z.reshape(PP.shape)
#    
#    
#    graphPlotter.pcolormesh(XX, PP, Z > 0, cmap=color_map)
#    graphPlotter.contour(XX, PP, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
#            levels=[-.5, 0, .5])
#            
#    graphPlotter.title(kernel)
#    graphPlotter.plot(x, y)
#    graphPlotter.show()
#
#
## Here you can output which ever result you would like by changing the Kernel and clf.predict lines
## Change kernel here to poly, rbf or linear
## adjusting the gamma level also changes the degree to which the model is fitted
#classifier = svm.SVC(kernel='poly', gamma=3).fit(X_train, y_train)
#
#
## Change the interger values within x.ix[:,[6,3]].dropna() explore the relationships between other
## features. the ints are column postions. ie. [6,3] 6th column and the third column are evaluated.
#res_svm = classifier.predict(x.ix[:,[6,3]].dropna())
#
#res_svm = DataFrame(res_svm,columns=['Trustworthy'])
#res_svm.to_csv("/Users/Kaushy/Courses/Project/SupportVectorMachine/Results.csv") # saves the results for you, change the name as you please.
#
#print pandas.notnull(twitterDataFrame['InReplyToStatusID']).values

#Dropping the First Header in the Row
#twitterData = twitterData[1:]
#NumberOfRowsOfData, NumberOfColumnsOfFeatures = twitterData.shape
#
## the target classification is the First column
#TargetClassificationRow = 39
#NumberOfColumnsOfFeatures = NumberOfColumnsOfFeatures-1
#
## Test Using only the first two features (Slicing in python 0,1,2,3,4,5 etc when you say 1:3 that is 1 and 2 in this case. So not 0 and not 3)
#XValueColumnVector = twitterData[:,:1]
#YValueColumnVector = twitterData['Classifier']
#
## Create a 1D array off the column vector we get from YValue
##YValue = YValueColumnVector.ravel()
##XValue = XValueColumnVector.ravel()
#
#print XValueColumnVector.shape
#print YValueColumnVector.shape
## Fit model to this and get the hyperplane. SVC is Support Vector Classification. Can also chose kernla to be poly, rbf, sigmoid, precomputed or callable depending on the shape of the
#myClassifier = svm.SVC(kernel='linear')
#myClassifier.fit(XValueColumnVector, YValueColumnVector)
##
##Getting the seperating Hyper Planes
#avgDifferenceBetweenTwoPoints = myClassifier.coef_[0]
## Slope or Gradient m = d(y) / d(x)
#a = -avgDifferenceBetweenTwoPoints[1] / avgDifferenceBetweenTwoPoints[0]
#xPlane = numpy.linspace(-40, 100)
#yPlane = a * xPlane - myClassifier.intercept_[0] / avgDifferenceBetweenTwoPoints[0] # y = mx + c
#
##The following commands bring up the cooridnates of the two planes.
#print xPlane
#print yPlane
#
## get the separating hyperplane using weighted classes
#weightedmyClassifier = svm.SVC(kernel='linear', class_weight={0: 20})
#weightedmyClassifier.fit(XValue, YValue)
#
#weightedavgDifferenceBetweenTwoPoints = weightedmyClassifier.coef_[0]
#weightedA = -weightedavgDifferenceBetweenTwoPoints[1] / weightedavgDifferenceBetweenTwoPoints[0]
#weightedyPlane = weightedA * xPlane - weightedmyClassifier.intercept_[0] / weightedavgDifferenceBetweenTwoPoints[1]
#
## plot separating hyperplanes
#hyperPlaneWithNoWeights = pythonLab.plot(xPlane, yPlane, '-r', label='no weights')
#hyperPlaneWithWeights = pythonLab.plot(xPlane, weightedyPlane, '--b', label='with weights')
#
## here c is color and cmap is color map. -g is solid green line. --b is double dash blue line and we are getting the first and second dimensions of the XValue variable which we defined
## at the top. So XValue 0 would be technically the second column and XValue 1 would be the third but anws
#pythonLab.scatter(XValue[:, 0], XValue[:, 1], c=YValue, cmap=pythonLab.cm.Paired)
#pythonLab.legend()
#
#pythonLab.axis('tight')
#pythonLab.show()
