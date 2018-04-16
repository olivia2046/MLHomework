import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys, os
script_path = os.path.split(os.path.realpath(__file__))[0] # 获取plotting.py脚本所在路径
#print ("current path: %s"%script_path)
sys.path.append(r"script_path\..")
import lib.calc as calc

def plotDataPoints(X, idx, K):
    #PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    #index assignments in idx have the same color
    #   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
    #   with the same index assignments in idx have the same color

    # 忽略原MATLAB程序的实现细节
    print(X.shape)
    # Plot the data
    plt.scatter(X[:,0], X[:,1], s=15, c=idx.ravel());
    #plt.show()
    # not using K?
    
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
    
    '''
    #test for adding text label
    for i in range(len(X)):
        plt.text(X[i][0],X[i][1],y[i])'''
    
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
        #plot_y = (-1/theta[2]).dot(theta[1].dot(plot_x) + theta[0])
        plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0]); #此处theta[0],theta[1],theta[2]都是标量值，因此是element-wise multiply
        

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
    '''print ("display_rows:%s"%display_rows)
    print ("display_cols:%s"%display_cols)
    print ("example_height:%s"%example_height)
    print ("example_width:%s"%example_width)'''
    
    # Between images padding
    pad = 1;

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad), \
                           pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0;
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break; 
            
            # Copy the patch
            
            # Get the max value of the patch
            max_val = np.max(np.absolute(X[curr_ex, :]))
            rowidx_start = pad + j * (example_height + pad)
            colidx_start = pad + i * (example_width + pad)
            #print(rowidx_start)
            #print(colidx_start)
            np.reshape(X[curr_ex, :], (example_height, example_width))/max_val
            display_array[rowidx_start:rowidx_start + example_height, \
                          colidx_start:colidx_start + example_width] = \
                            np.reshape(X[curr_ex, :], (example_height, example_width),order = 'F') / max_val 
                            #MATLAB默认按列填充，如果不加order='F',显示的图片90度翻转加镜像反转
            curr_ex = curr_ex + 1;
        
        if curr_ex > m:
            break; 
        
    # Display Image
    #h = plt.imshow(display_array, extent = [-1, 1]);
    h = plt.imshow(display_array,cmap = plt.get_cmap("gray")) # 如使用'Greys'会是反转片效果
    #h = plt.imshow(display_array)
    #h = plt.imshow(display_array,cmap=matplotlib.cm.binary)
    
    
    
    # Do not show axis
    #axis image off
    plt.axis("off")
    
    #plt.show()
    
def drawLine(p1, p2, *args, **kwargs):
    #DRAWLINE Draws a line from point p1 to point p2
    #   DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
    #   current figure

    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)