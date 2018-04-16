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
            #out[:,end] = (X1**(i-j))*(X2**j);
            
            temp = (X1**(i-j))*(X2**j)
            out = np.hstack((out, temp))
    
    return out
    
def featureNormalize(X):
    #FEATURENORMALIZE Normalizes the features in X 
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.

    mu = np.mean(X, axis =0); # mu should be a vector
    #X_norm = bsxfun(@minus, X, mu); 
    X_norm = X-mu

    sigma = np.std(X_norm, axis = 0);
    #X_norm = bsxfun(@rdivide, X_norm, sigma);
    X_norm = X_norm / sigma

    return X_norm, mu, sigma
    
def magic(n):
    row,col=0,n//2
    magic=[]
    for i in range(n):  
        magic.append([0]*n)
    magic[row][col]=1  
    for i in range(2,n*n+1):  
        r,l=(row-1+n)%n,(col+1)%n      
        if(magic[r][l]==0):
            row,col=r,l         
        else: 
            row=(row+1)%n  
        magic[row][col]=i
    marray = np.array(magic)
    return marray