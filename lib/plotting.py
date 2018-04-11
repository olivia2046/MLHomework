import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys, os
script_path = os.path.split(os.path.realpath(__file__))[0] # 获取plotting.py脚本所在路径
#print ("current path: %s"%script_path)
sys.path.append(r"script_path\..")
import lib.calc as calc


def plotData(X, y):
    #PLOTDATA Plots the data points X and y into a new figure 
    #   PLOTDATA(x,y) plots the data points with + for the positive examples

    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    #由于plotData()函数需要被plotDecisionBoundary()函数调用，因此输入参数应为numpy的ndarray而不是pandas的DataFrame


    # Create New Figure
    #figure; hold on;
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
        #plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
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
                z[i,j] = calc.mapFeature(u[i], v[j]).dot(theta)
        
        #print('z:')
        #print(z)
        #print('----------------------------')
        z = z.T; # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0] #what does [0,1] mean?
        #print('z.T:')
        #print(z)
        plt.contour(u, v, z, np.array([0,1]), linewidth=2)
        
def plot_an_image(image):
#     """
#     image : (400,)
#     """
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))
    plt.show()
#绘图函数

def displayData(X, example_width = None):
    #DISPLAYDATA Display 2D data in a nice grid
    #   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the 
    #   displayed array if requested.

    # Set example_width automatically if not passed in
    if example_width is None:
        #print(X.shape)
        example_width = int(np.round(np.sqrt(X.shape[1])))

    m, n = X.shape
    print("m, n: %s, %s"%(m,n))
    example_height = int(n / example_width);
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    print ("display_rows:%s"%display_rows)
    print ("display_cols:%s"%display_cols)
    """ sample 100 image and show them
    assume the image is square

    X : (5000, 400)
    """

    # sample 100 image, reshape, reorg it
    #sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100*400
    #sample_images = X[sample_idx, :]
    sample_images = X

    fig, ax_array = plt.subplots(nrows=display_rows, ncols=display_cols, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(display_rows):
        for c in range(display_cols):
            sample_images[display_cols * r + c]
            ax_array[r, c].matshow(sample_images[display_cols * r + c].reshape((example_width, example_height)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))  
            #绘图函数，画100张图片
            
    plt.show()