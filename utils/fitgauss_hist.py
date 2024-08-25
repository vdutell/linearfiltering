import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#fit guassian to histogram
def fitgauss_hist(datavec, nbins = 1000):
    
    datavec = datavec.flatten()
    
    #fit to a log-linear curve
    
    bin_heights, bin_borders, _ = plt.hist(datavec, bins=nbins, label='data')
    bin_centers = (bin_borders[:-1] + bin_borders[1:]) / 2 #fit to centers of bins, not edges

#     Unnormalized Gaussian
#     mu, sig = stats.norm.fit(datavec)
#     print(mu,sig)
#     pdf_fit = stats.norm.pdf(bin_centers, mu, sig)
#     plt.plot(bin_centers, pdf_fit ,label='Gaussian: $\mu$={0:0.2f}, $\sigma$={1:0.2f}'.format(mu,sig))


    #Gaussian with support for skewness
    def gauss_skew(x, amp, mu, sigma, alpha, c):
        #normal distribution
        normpdf = (1/(sigma*np.sqrt(2*math.pi)))*np.exp(-(np.power((x-mu),2)/(2*np.power(sigma,2))))
        normcdf = (0.5*(1+sp.erf((alpha*((x-mu)/sigma))/(np.sqrt(2)))))
        return (2*amp*normpdf*normcdf + c) * np.sign(sigma)
    #make an initial guess & fit
    init_guess = [np.max(bin_heights),np.mean(datavec), np.var(datavec), 1, 1]
    fitparms, fitcov = curve_fit(gauss_skew, bin_centers, bin_heights, p0=init_guess)
    #plot
    plt.plot(bin_centers, gauss_skew(bin_centers, *fitparms),
             label='SkewGauss: $\mu={0:0.2f}, \sigma={1:0.2f}, \delta={2:0.2f}$'.format(fitparms[1],fitparms[2],fitparms[3]))

#   vanilla gaussian distribution
    def gaus(x,amp,mu,sigma):
        #multiply by sign of sigma to make sure we get a positive value
        return amp*np.exp(-(x-mu)**2/(2*sigma**2))*np.sign(sigma)
    #make an initial guess & fit
    init_guess = [np.max(bin_heights),np.mean(datavec), np.var(datavec)]
    fitparms, fitcov = curve_fit(gaus, bin_centers, bin_heights, p0 = init_guess)
    #plot
    plt.plot(bin_centers, gaus(bin_centers, *fitparms),label='Gauss: $\mu={0:0.2f}, \sigma={1:0.2f}$'.format(fitparms[1],fitparms[2]))


    plt.legend(fontsize=6)
