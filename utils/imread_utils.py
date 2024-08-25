import cv2
import numpy as np

import utils.distributions as dsts

def readin_im(filename, imfolder):
    im = cv2.imread(imfolder+filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) #fix colors 
    return(im)

def cosine_window(image):
    #raised cosyne window on image to avoid border artifacts
    (dim1,dim2) = np.shape(image)
    oneD = np.hanning(dim1)
    oneD = np.tile(oneD,(dim2,1)).T
    twoD = np.hanning(dim2)
    twoD = np.tile(twoD,(dim1,1))
    cosfilter = oneD*(twoD) 
    
    filtered_im = image * cosfilter
    return(filtered_im)

def read_proc_im(filename, imfolder, rescale=True, mean=False, cosine=False):
    #read image in
    im = readin_im(filename, imfolder)
    # average over color channel
    im = np.mean(im,axis=-1)
    #subtract mean
    if(mean):
        im = im - np.mean(im)
    #rescale between 0 and 1 
    if(rescale):
        im = dsts.rescale(im)
    #take cosine window
    if(cosine):
        im = cosine_window(im)
    #remove dc component as last step before returning
    im = dsts.rmdc(im)
    return(im)

def cropim(im, cropx, cropy):    
    y,x = im.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    if((cropx <= x) & (cropy <=y)):
        return im[starty:starty+cropy,startx:startx+cropx]
    else:
        return(float('nan'))

def readpreprc_ims(filenames, imfolder, rescale=True, mean=False, cosine=False, crop=(512,512)):

    ims = [read_proc_im(fname, imfolder, rescale, mean, cosine) for fname in filenames]
    
    if(crop):
        #mnx = min([np.shape(a)[1] for a in ims])
        #mny =min([np.shape(a)[0] for a in ims])
        ims = [cropim(im, *crop) for im in ims]
        ims = [im for im in ims if ~np.isnan(np.mean(im))]
        #print(ims[1:10])
        ims = np.array(ims)

    ims = np.array(ims) #cast to array
    return(ims)
