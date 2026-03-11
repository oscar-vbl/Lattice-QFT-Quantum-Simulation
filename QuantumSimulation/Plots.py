import matplotlib.pyplot as plt

def simplePlot(x, y, title="", xlabel="", ylabel="", savePath=None):
    '''
    Basic plot function for testing
    '''
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if savePath:
        plt.savefig(savePath)
    return fig, ax
