import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
    example_height = int(n / example_width);
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    """ sample 100 image and show them
    assume the image is square

    X : (5000, 400)
    """

    # sample 100 image, reshape, reorg it
    #sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100*400
    #sample_images = X[sample_idx, :]
    sample_images = X

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(display_rows):
        for c in range(display_cols):
         
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((example_width, example_height)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))  
            #绘图函数，画100张图片
            
    plt.show()