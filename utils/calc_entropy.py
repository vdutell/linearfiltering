import numpy as np
def calc_entropy(im, rescale=True):
    
    im = im.flatten()

    #bin 
    counts,boundaries = np.histogram(im)
    centers = (boundaries[1:] + boundaries[:-1])/2

    vals_prob = counts.astype(np.float) / np.sum(counts)

    #calc entropy contribution for each instance
    #some values are zero because they have probability zero. Make these zero
    vals_entropy_contrib = -1 * vals_prob * np.nan_to_num(np.log(vals_prob)/np.log(2))

    #calc entropy total
    entropy = np.sum(vals_entropy_contrib)

    return(entropy)
