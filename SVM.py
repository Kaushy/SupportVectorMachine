from sklearn import svm
from DataManipulation import X_train,y_train, y, x,X_test,y_test
import pylab as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    
    try:
        X, y = X.values, y.values
    except AttributeError:
        pass
    
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
                         
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
                         
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()


def SupportVectorMachineImplementation():
    svc_rbf = svm.SVC(kernel='rbf', gamma=1e2)
    svc_rbf.fit(X_train, y_train)
    print svc_rbf.score(X_test, y_test)
    #plot_estimator(svc_rbf, X_train, y_train)
    #plt.scatter(svc_rbf.support_vectors_[:, 0], svc_rbf.support_vectors_[:, 1],
    #s=80, facecolors='none', linewidths=2, zorder=10)
    #plt.title('RBF kernel')
#plt.show();