print(__doc__)

import numpy as numpy
import sys
import  matplotlib
import pylab as pythonLab
import pandas as pandas
from sklearn import svm
import matplotlib.pyplot as graphPlotter
from sklearn.linear_model import LogisticRegression
from patsy import dmatrices
from utility import plot_decision_boundary
from sklearn.utils import shuffle

DataSetLocation = r'/Users/Kaushy/Courses/Project/SupportVectorMachine/FinalFirst.csv'
DataSetLocationUserData = r'/Users/Kaushy/Courses/Project/SupportVectorMachine/UserAccountInfoFinal.csv'
numpy.set_printoptions(threshold=sys.maxsize)

# Option to display more data when printed on console
pandas.set_option('display.max_rows', 50000)
pandas.set_option('display.max_columns', 60000)

# ---------------------------------------------------------------------- TwitterData ------------------------------------------------------------------------ #
twitterDataFrame = pandas.read_csv(DataSetLocation)
dropColumnList = ['ID','Source','Truncated','InReplyToScreenName','UserScreenName','GEO','Coordinates','PlaceID','PlaceURL','PlaceType','PlaceName','PlaceFullName','PlaceCountryCode','PlaceCountry','URLInTweetExpanded','Contributors','Symbols','Indices','MediaURLShort','MediaIndices','UserMentionScreenName','UserMentionName','Favourited','Retweeted']
twitterDataFrame = twitterDataFrame.drop(dropColumnList,axis=1)

# Modifying our dataset with relevant values to fit our analysis needs. The following is taken as it is : TweetLength, Language
twitterDataFrame['CreatedAt'] = twitterDataFrame['CreatedAt'].map(lambda x: pandas.to_datetime(x,dayfirst=True))
twitterDataFrame['CreatedAtForCalculations'] = twitterDataFrame['CreatedAt']
twitterDataFrame['InReplyToStatusID'] = twitterDataFrame['InReplyToStatusID'].map(lambda x: True if pandas.notnull(x) else False)
twitterDataFrame['InReplyToUserID'] = twitterDataFrame['InReplyToUserID'].map(lambda x: True if pandas.notnull(x) else False)
twitterDataFrame['RetweetCount'] = twitterDataFrame['RetweetCount'].map(lambda x: x if pandas.notnull(x) else 0)
twitterDataFrame['FavouriteCount'] = twitterDataFrame['FavouriteCount'].map(lambda x: x if pandas.notnull(x) else 0)
twitterDataFrame['Hashtags'] = twitterDataFrame['Hashtags'].map(lambda x: True if pandas.notnull(x) else False)
twitterDataFrame['URL'] = twitterDataFrame['URL'].map(lambda x: True if pandas.notnull(x) else False)
twitterDataFrame['MediaURL'] = twitterDataFrame['MediaURL'].map(lambda x: True if pandas.notnull(x) else False)
twitterDataFrame['MediaType'] = twitterDataFrame['MediaType'].map(lambda x: x if pandas.notnull(x) else False)
twitterDataFrame['UserMentionID'] = twitterDataFrame['UserMentionID'].map(lambda x: True if pandas.notnull(x) else False)
twitterDataFrame['PossiblySensitive'] = twitterDataFrame['PossiblySensitive'].map(lambda x: x if pandas.notnull(x) else 'NoData')
twitterDataFrame['TweetLength'] = twitterDataFrame['TweetLength'].map(lambda x: x if pandas.notnull(x) else 0)

## ------------------------------------------------ Force Columns to be one dtype ------------------------------------------------ ##
twitterDataFrame['CreatedAt'] = twitterDataFrame['CreatedAt'].convert_objects(convert_dates='coerce')
twitterDataFrame['CreatedAtForCalculations'] = twitterDataFrame['CreatedAtForCalculations'].convert_objects(convert_dates='coerce')
twitterDataFrame['RetweetCount'] = twitterDataFrame['RetweetCount'].convert_objects(convert_numeric='coerce')
twitterDataFrame['FavouriteCount'] = twitterDataFrame['FavouriteCount'].convert_objects(convert_numeric='coerce')
twitterDataFrame['UserID'] = twitterDataFrame['UserID'].convert_objects(convert_numeric='coerce')
twitterDataFrame['Classifier'] = twitterDataFrame['Classifier'].convert_objects(convert_numeric='coerce')
twitterDataFrame['TweetLength'] = twitterDataFrame['TweetLength'].convert_objects(convert_numeric='coerce')

### ------------------- Calculating Values for Deriving New Features for our DataFrame ------------------- ###
# Frequency of Tweets
twitterDataFrame = twitterDataFrame.set_index(['CreatedAt'])
tweetsByEachUser = twitterDataFrame.groupby('UserID')
numberOfHoursBetweenFirstAndLastTweet = abs(tweetsByEachUser['CreatedAtForCalculations'].last() - tweetsByEachUser['CreatedAtForCalculations'].first()).astype('timedelta64[h]')

numberOfTweetsByTheUser = tweetsByEachUser.size()
frequency = numberOfTweetsByTheUser  / numberOfHoursBetweenFirstAndLastTweet
getFrequency = lambda x : numberOfTweetsByTheUser  / numberOfHoursBetweenFirstAndLastTweet
twitterDataFrame = pandas.merge(twitterDataFrame.reset_index(),frequency.reset_index(),on=['UserID'],how='inner').set_index(['CreatedAt'])
twitterDataFrame = twitterDataFrame.rename(columns={0 : 'FrequencyOfTweets'})

# ---------------------------------------------------------------------- TwitterUserData --------------------------------------------------------------------- #
twitterUserDataFrame = pandas.read_csv(DataSetLocationUserData,index_col='UserID')
dropUserColumnList = ['Name','ScreenName','EntityURL','EntityExpandedURL','Indices','DescriptionURL','DescriptionExpandedURL','DescriptionIndicesURL','ShowAllInlineMedia']
twitterUserDataFrame = twitterUserDataFrame.drop(dropUserColumnList,axis=1)

# Modifying our dataset with relevant values to fit our analysis needs
twitterUserDataFrame['Description'] = twitterUserDataFrame['Description'].map(lambda x: len(x) if pandas.notnull(x) else 0)
twitterUserDataFrame['UserAccountURL'] = twitterUserDataFrame['UserAccountURL'].map(lambda x: True if pandas.notnull(x) else False)
twitterUserDataFrame['UserAccountCreatedAt'] = twitterUserDataFrame['UserAccountCreatedAt'].map(lambda x: pandas.to_datetime(x))
twitterUserDataFrame['ProfileBackgroundImageURL'] = twitterUserDataFrame['ProfileBackgroundImageURL'].map(lambda x: True if pandas.notnull(x) else False)
twitterUserDataFrame['ProfileImageURL'] = twitterUserDataFrame['ProfileImageURL'].map(lambda x: True if pandas.notnull(x) else False)
twitterUserDataFrame['ProfileBannerURL'] = twitterUserDataFrame['ProfileBannerURL'].map(lambda x: True if pandas.notnull(x) else False)

# Merging the two dataframes - user and the tweets
finalDataFrame = pandas.merge(twitterDataFrame.reset_index(),twitterUserDataFrame.reset_index(),on=['UserID'],how='inner')
finalDataFrame = finalDataFrame.drop_duplicates()
finalDataFrame['FrequencyOfTweets'] = numpy.all(numpy.isfinite(finalDataFrame['FrequencyOfTweets']))
print finalDataFrame.info()


# model formula, ~ means = and C() lets the classifier know its categorical data.
formula = 'Classifier ~ InReplyToStatusID + InReplyToUserID + RetweetCount + FavouriteCount + Hashtags + UserMentionID + URL + MediaURL + C(MediaType) + UserMentionID + C(PossiblySensitive) + C(Language) + TweetLength + Location + Description + UserAccountURL + Protected + FollowersCount + FriendsCount + ListedCount + UserAccountCreatedAt + FavouritesCount + GeoEnabled + StatusesCount + ProfileBackgroundImageURL + ProfileUseBackgroundImage + DefaultProfile + FrequencyOfTweets'

### create a regression friendly data frame y gives the classifiers, x gives the features and gives different columns for Categorical data depending on variables. 
y, x = dmatrices(formula, data=finalDataFrame, return_type='matrix')

## select which features we would like to analyze
X = numpy.asarray(x)

X = X[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95]]

# needs to be 1 dimenstional so we flatten. it comes out of dmatirces with a shape.
y = y.ravel()
n_sample = len(X)

# Split the dataset into a training set and a test set.
sixtyPercentOfSample = int(.6 * n_sample)
X, y = shuffle(X, y)
X_train = X[:sixtyPercentOfSample]
y_train = y[:sixtyPercentOfSample]
X_test = X[sixtyPercentOfSample:]
y_test = y[sixtyPercentOfSample:]


graphPlotter.prism()
graphPlotter.scatter(X_train[:, 0], X_train[:, 1], c=y)
graphPlotter.show()

graphPlotter.prism()
graphPlotter.scatter(X_train[:, 0], X_train[:, 1],X_test[:, 2], c=y_train)
graphPlotter.scatter(X_test[:, 0], X_test[:, 1],X_test[:, 2], c='white', marker='^')
#graphPlotter.show()


## ------------ Data Pruning ----------------##
# Import logistic regression from scikit-learn and generate a classification object.
logreg = LogisticRegression(class_weight='auto')

#Now let's fit the logistic regression model to the training data, to see if we can do it automatically:
logreg.fit(X_train, y_train)

print "Result Parameters : ", logreg.get_params(deep='True')
print "Confidence Scores : ", logreg.decision_function(X_train)
#
# First, we have a look at how logistic regression did on the training set.
#We do this by now setting colors using the predicted class.

y_pred_train = logreg.predict(X_train)
graphPlotter.scatter(X_train[:, 0], X_train[:, 1],X_train[:, 2], c=y_pred_train)
plot_decision_boundary(logreg, X)
#graphPlotter.show()

print "Accuracy on training set:", logreg.score(X_train, y_train)

y_pred_test = logreg.predict(X_test)
graphPlotter.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, marker='^')
graphPlotter.scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train)
plot_decision_boundary(logreg, X)
#graphPlotter.show()

print "Accuracy on test set:", logreg.score(X_test, y_test)

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
