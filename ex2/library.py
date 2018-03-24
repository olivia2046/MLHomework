import matplotlib.pyplot as plt
import numpy as np



def mapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features
    #
    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features used in the regularization exercise.
    #
    #   Returns a new feature array with more features, comprising of 
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size
    
    # 输入数组，转换为两个列向量
    X1 = X1.reshape(-1,1)
    X2 = X2.reshape(-1,1)

    degree = 6;
    #end = 0
    #out = np.ones((len(X1[:,0]),1));
    out = np.ones((len(X1),1));#需要同时处理一维和二维数组
    for i in range(1,degree+1,1):
        for j in range (i+1):
           
            temp = (X1**(i-j))*(X2**j)
            out = np.hstack((out, temp))
    
    return out
    

def plotData(X, y):
    #PLOTDATA Plots the data points X and y into a new figure 
    #   PLOTDATA(x,y) plots the data points with + for the positive examples

    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    #由于plotData()函数需要被plotDecisionBoundary()函数调用，因此输入参数应为numpy的ndarray而不是pandas的DataFrame


    # Create New Figure
    fig = plt.figure()

    # Instructions: Plot the positive and negative examples on a
    #               2D plot, using the option 'k+' for the positive
    #               examples and 'ko' for the negative examples.
    #

    # Find Indices of Positive and Negative Examples
    pos = np.where(y==1)
    neg = np.where(y==0)
    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k+',linewidth=2, markersize=7);
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7);
    
    #plt.show()
    
    
def plotDecisionBoundary(theta, X, y):
    #PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    #the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
    #   positive examples and o for the negative examples. X is assumed to be 
    #   a either 
    #   1) Mx3 matrix, where the first column is an all-ones column for the 
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones

    # Plot Data
    plotData(X[:,1:], y);
    #hold on

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X[:,1])-2,  max(X[:,1])+2];

        # Calculate the decision boundary line
        #决策边界：θ0+θ1x1+θ2x2=0
        plot_y = np.dot((-1/theta[2]),(np.dot(theta[1], plot_x) + theta[0]));

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50);
        v = np.linspace(-1, 1.5, 50);

        z = np.zeros((len(u), len(v)));
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = mapFeature(u[i], v[j]).dot(theta)
        
        #print('z:')
        #print(z)
        #print('----------------------------')
        z = z.T; # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0] #what does [0,1] mean?
        #print('z.T:')
        #print(z)
        #plt.contour(u, v, z, np.array([0,1]), linewidth=2) #used by contour: 'linewidth'
        plt.contour(u, v, z, np.array([0,1]))
    

def predict(theta, X):
    #PREDICT Predict whether the label is 0 or 1 using learned logistic 
    #regression parameters theta
    #   p = PREDICT(theta, X) computes the predictions for X using a 
    #   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    m = len(X); # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros((m, 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned logistic regression parameters. 
    #               You should set p to a vector of 0's and 1's
    #
    
    p = sigmoid(np.dot(X,theta))
    p = np.array([pi >=0.5 for pi in p]).reshape(-1,1)#有没有更好的写法？

    # =========================================================================
    return p
    