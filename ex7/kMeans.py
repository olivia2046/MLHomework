import numpy as np
import sys
sys.path.append('..')
import lib.plotting as libplt

def computeCentroids(X, idx, K):
    #COMPUTECENTROIDS returns the new centroids by computing the means of the 
    #data points assigned to each centroid.
    #   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
    #   computing the means of the data points assigned to each centroid. It is
    #   given a dataset X where each row is a single data point, a vector
    #   idx of centroid assignments (i.e. each entry in range [1..K]) for each
    #   example, and K, the number of centroids. You should return a matrix
    #   centroids, where each row of centroids is the mean of the data points
    #   assigned to it.
    #

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))


    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every centroid and compute mean of all points that
    #               belong to it. Concretely, the row vector centroids(i, :)
    #               should contain the mean of the data points assigned to
    #               centroid i.
    #
    # Note: You can use a for-loop over the centroids to compute this.
    #
    '''for k in range(K):
        centroids[k] = np.average(X[(idx==k).ravel()],axis=0)'''
    # no loop
    centroids = np.array([np.average(X[(idx==k).ravel()],axis=0) for k in range(K)])
    # =============================================================

    return centroids

def plotProgresskMeans(X, centroids, previous, idx, K, i):
    #PLOTPROGRESSKMEANS is a helper function that displays the progress of 
    #k-Means as it is running. It is intended for use only with 2D data.
    #   PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
    #   points with colors assigned to each centroid. With the previous
    #   centroids, it also plots a line between the previous locations and
    #   current locations of the centroids.
    #

    # Plot the examples
    libplt.plotDataPoints(X, idx, K);

    # Plot the centroids as black x's
    plt.plot(centroids[:,0].ravel(), centroids[:,1].ravel(), 'x', \
         markeredgecolor='k', \
         markersize=10, linewidth=3);

    # Plot the history of the centroids with lines
    for j in range(len(centroids)):
        libplt.drawLine(centroids[j, :], previous[j, :])
    

    # Title
    plt.title('Iteration number %d'%i)
    
    plt.show()

def findClosestCentroids(X, centroids):
    #FINDCLOSESTCENTROIDS computes the centroid memberships for every example
    #   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
    #   in idx for a dataset X where each row is a single example. idx = m x 1 
    #   vector of centroid assignments (i.e. each entry in range [1..K])
    #

    # Set K
    K = len(centroids)

    # You need to return the following variables correctly.
    idx = np.zeros((len(X), 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every example, find its closest centroid, and store
    #               the index inside idx at the appropriate location.
    #               Concretely, idx(i) should contain the index of the centroid
    #               closest to example i. Hence, it should be a value in the 
    #               range 1..K
    #
    # Note: You can use a for-loop over the examples to compute this.
    #
    '''for i in range(len(X)):
        idx[i] = np.argmin([np.linalg.norm(X[i]-cent)**2 for cent in centroids])'''
    #no loop
    idx = np.array([np.argmin([np.linalg.norm(x-cent)**2 for cent in centroids]) for x in X]).reshape(-1,1)
    # =============================================================

    return idx

def runkMeans(X, initial_centroids, max_iters, plot_progress = False):
    #RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    #is a single example
    #   [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
    #   plot_progress) runs the K-Means algorithm on data matrix X, where each 
    #   row of X is a single example. It uses initial_centroids used as the
    #   initial centroids. max_iters specifies the total number of interactions 
    #   of K-Means to execute. plot_progress is a true/false flag that 
    #   indicates if the function should also plot its progress as the 
    #   learning happens. This is set to false by default. runkMeans returns 
    #   centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
    #   vector of centroid assignments (i.e. each entry in range [1..K])
    #

    # Initialize values
    m, n = X.shape
    K = len(initial_centroids)
    centroids = initial_centroids;
    previous_centroids = centroids;

    idx = np.zeros((m, 1))

    # Run K-Means
    for i in range(1,max_iters+1):
        
        # Output progress
        print('K-Means iteration %d/%d...\n'%(i, max_iters))

        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)
        
        #第一个迭代，质心没有发生位移，因此是在更新质心前输出图像
        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids;
        
        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)
        
        '''print(previous_centroids)
        print(centroids)'''

        # 判断质心位置是否改变
        #if (centroids==previous_centroids).all: #错误
        if np.all(centroids==previous_centroids):
            break;
        
    
    return centroids, idx
    
def kMeansInitCentroids(X, K):
    #KMEANSINITCENTROIDS This function initializes K centroids that are to be 
    #used in K-Means on the dataset X
    #   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
    #   used with the K-Means on the dataset X
    #

    # You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X
    #
    # Initialize the centroids to be random examples
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(len(X))
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]
    # =============================================================

    return centroids