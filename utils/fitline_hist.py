import numpy as np
import matplotlib.pyplot as plt

#fit data to a line and plot
def fitline_hist(datavec, nbins = 1000):
    
    #fit to a log-linear curve
    
    bin_heights, bin_borders, _ = plt.hist(datavec.flatten(), bins=nbins)
    bin_centers = (bin_borders[:-1] + bin_borders[1:]) / 2 #fit to centers of bins, not edges

    fit, resid, rank, sv, rcond = np.polyfit(x=bin_centers, y=np.log(1+bin_heights), deg=1, full=True)
    m,b = fit
    resid = resid[0]
    plt.semilogy(bin_centers,
                 np.exp(m*bin_centers+ b) - 1,
                 label='fit = exp({0:0.1f}x + {1:0.1f})-1, R={2:0.1f}'.format(m, b, resid))
    plt.legend()
    plt.title('M = {0:0.1f}, B = {1:0.1f}, Residual={2:0.1f}'.format(m,b,resid))
