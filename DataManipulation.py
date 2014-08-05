import numpy as numpy
import sys
import matplotlib
import pylab as pythonLab
import pandas as pandas
import matplotlib.pyplot as graphPlotter

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
print numberOfTweetsByTheUser
frequency = numberOfTweetsByTheUser  / numberOfHoursBetweenFirstAndLastTweet
getFrequency = lambda x : numberOfTweetsByTheUser  / numberOfHoursBetweenFirstAndLastTweet
twitterDataFrame = pandas.merge(twitterDataFrame.reset_index(),frequency.reset_index(),on=['UserID'],how='inner').set_index(['CreatedAt'])
twitterDataFrame = twitterDataFrame.rename(columns={0 : 'FrequencyOfTweets'})
print twitterDataFrame['FrequencyOfTweets']


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
twitterUserDataFrame['UserAccountCreatedAt'] = twitterUserDataFrame['UserAccountCreatedAt'].convert_objects(convert_dates='coerce')

# Merging the two dataframes - user and the tweets
finalDataFrame = pandas.merge(twitterDataFrame.reset_index(),twitterUserDataFrame.reset_index(),on=['UserID'],how='inner')
finalDataFrame = finalDataFrame.drop_duplicates()



# model formula, ~ means = and C() lets the classifier know its categorical data.
formula = 'Classifier ~ InReplyToStatusID + InReplyToUserID + RetweetCount + FavouriteCount + Hashtags + UserMentionID + URL + MediaURL + C(MediaType) + UserMentionID + C(PossiblySensitive) + C(Language) + TweetLength + Location + Description + UserAccountURL + Protected + FollowersCount + FriendsCount + ListedCount + FavouritesCount + Verified + GeoEnabled + StatusesCount + ProfileBackgroundImageURL + ProfileUseBackgroundImage + DefaultProfile + FrequencyOfTweets'

### create a regression friendly data frame y gives the classifiers, x gives the features and gives different columns for Categorical data depending on variables.
y, x = dmatrices(formula, data=finalDataFrame, return_type='matrix')

## select which features we would like to analyze
X = numpy.asarray(x)

X = X[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]]

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
