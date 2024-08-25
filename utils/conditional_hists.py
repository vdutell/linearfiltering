import numpy as np
import matplotlib.pyplot as plt
import pyrtools as pt


def butterfly_conditional(pyrimg1, pyrimg2, nbins=20, make_plot=True, save_plot_fname='', showfig=True, show_cbar=True, show_labels=True,
                          title_label='', band1_label='', band2_label='', subband_1_bins=None, subband_2_bins=None, column_normalize=True, pdist_norm=False):
    '''
    Create conditional histogram from two pyramid images. Calculates array representing conditional histogram as well as plots it.
    Conditional histograms are plots such as in Schwartz & Simoncelli 2001, Figure 2. These represent the histogram of values in a pyramid image
    conditioned on values from another pyramid image. This shows positive/negative/lack-of correlation between pyramid images at matching spatial locations.
    
    Parameters:
    pyrimg1 (2d array): pyramid image (or any image really) which we will condition values in pyrimg2 ON. 
    pyrimg2 (2d arary): pyrmaid image (or any image really) which we will display distributions contingent on pyramid1.
    nbins (int): number of histogram bins (size of output image)
    make_plot (bool): Should we actually make a figure?
    save_plot_fname (str): If we make a plot whats its filepath? (If left as default '', plot is not saved'.
    showfig (bool): plotting show (must be false for averaging)
    subband_1_bins (1d array): Specify bin locations (used when averaging)
    subband_2_bins (1d array): Specify bin locations (used when averaging)
    column_normalize (bool): Should we normalize column values to be in range [0,1] (this should be True for a S&S2001 plot, but False for averaging because its done later).
    pdist_norm (bool): normalize columns such that total area =1 instead of max of 1.
    
    Returns:
    butterfly (2d array): Values of butterfly plot
    
    '''
    
    
    #if pyramid images are different sizes, need to upscale smaller one
    if pyrimg1.shape[0] > pyrimg2.shape[0]:
        n_levels =  np.log2(pyrimg1.shape[0] / pyrimg2.shape[0])
        pyrimg2 = pt.tools.convolutions.upBlur(pyrimg2, n_levels=n_levels)
    elif pyrimg1.shape[0] < pyrimg2.shape[0]:
        n_levels =  np.log2(pyrimg2.shape[0] / pyrimg1.shape[0])
        pyrimg1 = pt.tools.convolutions.upBlur(pyrimg1, n_levels=n_levels)

    #create histogram from filter imgs and get indices
    pyrimg_1_flat = pyrimg1.flatten()
    pyrimg_2_flat = pyrimg2.flatten()
    
    #create bins
    if(subband_1_bins is None):
        hist_1, bins_1 = np.histogram(pyrimg_1_flat,bins=nbins) #set bins for pyr1
        hist_2, bins_2 = np.histogram(pyrimg_2_flat,bins=nbins) #set bins for pyr2
    else:
        bins_1 = subband_1_bins
        bins_2 = subband_2_bins

    #loop through indices as rows in our plot
    butterfly = []
    for bin_idx, b in enumerate(bins_1[:-1]):
        #get start and end of bin values
        bin_min = bins_1[bin_idx]
        bin_max = bins_1[bin_idx+1]
        #grab the pixel indices 
        pixel_idxes = [pixel_idx for pixel_idx,p in enumerate(pyrimg_1_flat) if (p>=bin_min and p<bin_max)]
        #pull out values for pyramid 2 at these indices (aka conditioning on pyramid 1 value)
        row = pyrimg_2_flat[pixel_idxes]
        rowbins = np.digitize(row,bins_2)
        rowhist = [len(row[rowbins == j]) for j in range(1,len(bins_2))]
        if(column_normalize):
            if(pdist_norm):
                rowhist = rowhist/np.sum(rowhist) #normalize such that total area =1
            else:
                rowhist = rowhist/np.max(rowhist) #normalize fo fill range[0,1]
        butterfly.append(rowhist)
    #butterfly.append(np.zeros_like(rowhist)+1) #to double check direction
    butterfly = np.array(butterfly).T
    
    #make plot
    if(make_plot):
        #plt.pcolormesh(butterfly)  
        plt.pcolormesh(bins_1,
                       bins_2,
                       butterfly)
        if(show_labels):
            plt.xlabel(f'{band1_label} (bin value)')
            plt.ylabel(f'{band2_label} (bin value)')
        if(show_cbar):
            plt.colorbar(label='Subband 2 Linear Value (normalized)')
        plt.title(title_label)
        
        if(save_plot_fname != ''):
            plt.savefig(save_plot_fname)
            plt.clf()
        elif(showfig):
            plt.show()     
            
    #return array
    return(butterfly)


def butterfly_conditional_avg(subband1_list, subband2_list,
                              nbins=20,
                              make_plot=True,
                              save_plot_fname='',
                              showfig=False,
                              show_cbar=True,
                              show_labels=True,
                              title_label='',
                              band1_label='',
                              band2_label='',
                              post_avg_norm=True,
                              pdist_norm=False):
    '''
    Create a conditional histogram plot that is averaged over many image subbands
    
    Wrapper for butterfly_conditional() function allowing averaging over many image pyramids.
    
    Params:
    see buterfly_condititional
    post_avg_norm (bool): normalize columns to be in range [0,1] AFTER averaging over many conditional hists.
    '''
    
    #before calculating min and max, need to upsample to max and min are correctly calculated
    for i in range(len(subband1_list)):
        pyrimg1 = subband1_list[i]
        pyrimg2 = subband2_list[i]
        #if pyramid images are different sizes, need to upscale smaller one
        if pyrimg1.shape[0] > pyrimg2.shape[0]:
            n_levels =  np.log2(pyrimg1.shape[0] / pyrimg2.shape[0])
            pyrimg2 = pt.tools.convolutions.upBlur(pyrimg2, n_levels=n_levels)
        elif pyrimg1.shape[0] < pyrimg2.shape[0]:
            n_levels =  np.log2(pyrimg2.shape[0] / pyrimg1.shape[0])
            pyrimg1 = pt.tools.convolutions.upBlur(pyrimg1, n_levels=n_levels)
        #flatten and put back in list
        subband1_list[i] = pyrimg1.flatten()
        subband2_list[i] = pyrimg2.flatten()


    #first figure out min and max of band1 filters and band2 filters.
    band1_min = min([np.min(x) for x in subband1_list]) #note why do we need nan?
    band1_max = max([np.max(x) for x in subband1_list])
    subband_1_bins = np.linspace(band1_min, band1_max, nbins+1)
    band2_min = min([np.min(x) for x in subband2_list])
    band2_max = max([np.max(x) for x in subband2_list])
    subband_2_bins = np.linspace(band2_min, band2_max, nbins+1)

    chists_list = []
    for i in range(len(subband1_list)):
        chist = butterfly_conditional(subband1_list[i],
                                         subband2_list[i],
                                         nbins=nbins,
                                         make_plot=False,
                                         showfig=False,
                                         subband_1_bins=subband_1_bins,
                                         subband_2_bins=subband_2_bins,
                                         column_normalize=(not post_avg_norm) #if we want to normalize after averaging, then column_normalize is FALSE for individual hists
                                     )
        chists_list.append(chist)
        
    #average over histograms 
    chists_list = np.nanmean(chists_list,axis=0) #no longer a list but this saves memory
    
    #normaze AFTER averaging plots
    if(post_avg_norm):
        for i in range(np.shape(chists_list)[1]):
            col = chists_list[i,:]
            if(pdist_norm):
                col = col / np.sum(col) #normalize such that total area =1
            else:
                col = col / np.max(col) #histograms are positive integer count values so normalize by dividing by max
            chists_list[:,i] = col
            
    #make plot
    if(make_plot):
#        plt.pcolormesh(chists_list) # to test
        plt.pcolormesh(subband_1_bins,
                       subband_2_bins,
                       chists_list)
        if(show_labels):
            plt.xlabel(f'{band1_label} (bin value)')
            plt.ylabel(f'{band2_label} (bin value)')
            if(show_cbar):
                plt.colorbar(label='Subband 2 Value (normalized)')
            plt.title(title_label)

            if(save_plot_fname != ''):
                plt.savefig(save_plot_fname)
                plt.clf()
            elif(showfig):
                plt.show()     
            
    #return array
    print('*',end='')
    return(chists_list)  
