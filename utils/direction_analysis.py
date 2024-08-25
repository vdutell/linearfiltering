import numpy as np

def direction_analysis(signal2d, func, nbins=20, sample=True, center=None):
    """
    Calculate a signal along radial profile.

    signal2d - 2d signal to be analyzed (not necessarily an signal_2ds)
    func - function to be applied to signal radially
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the signal (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the signal_2ds
    y, x = np.indices(signal2d.shape)

    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    
    a = np.arctan2(( y - center[1]),(x - center[0]))
    #return(r) # can run imshow(r) to test correct

    ind = np.argsort(a.flat)
    a_sorted = a.flat[ind]
    i_sorted = signal2d.flat[ind]

    #digitize cleanly
    linbins = np.linspace(-np.pi,np.pi,nbins+1)
    a_digitized = np.take(linbins,np.digitize(a_sorted,linbins,right=True))

    #digitize by int
    # Get the integer part of the angle (bin size = 1)
    #a_digitized = a_sorted.astype(int)

    #if we are using a quantity that depends on population size, figure out the smallest population size in an angle
    #sample to reduce all angles to this number of samples.
    if(sample):
        from collections import Counter
        least_common = Counter(a_digitized).most_common()[-1]
        subset_size = least_common[1] #get number of values for least common

    angs = np.unique(a_digitized)
    vals= []
    for ang in angs:
        pxls = i_sorted[np.where(a_digitized == ang)]
        if(sample):
            pxls = np.random.choice(pxls, size=subset_size, replace=False) #take the smallest subset
        vals.append(func(pxls))

    return(angs, vals)
