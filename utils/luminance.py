import numpy as np

#helper function converting sRGB to linear space
def srgb2lin(s):
    limit = 0.0404482362771082 
    s[s <= limit] = s[s <= limit]/12.92
    s[s > limit] = np.power(((s[s>limit] + 0.055) / 1.055), 2.4)
    return s

#helper function undoing alpha correction
def lin2srgb(lin):
    limit = 0.0031308
    lin[lin > limit] = 1.055 * (np.power(lin[lin > limit], (1.0 / 2.4))) - 0.055
    lin[lin <= limit] = 12.92 * lin[lin <= limit]
    return lin


def calc_lum_img(img):
    '''
    Calculate a luminance image from an RGB image using the standard luminance transform:
    https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale
    0.2126R, 0.7152G, 0.0722B (where RGB are linear)
    
    '''
    #normalize image to [0,1] first if needed.
    #if largest number is >1 assume its [0,255]
    if(np.max(img)>=1):
        img = img / 255. 
    #if smallest number is negative, assume it's [-1,1]
    if(np.min(img)<0):
        img = img + 1.
        img = img / np.max(img)
    #linearize, apply sRGB -> luminance transform, then undo linearization
    img = srgb2lin(img)
    img = (0.2126*img[:,:,0])+(0.7152*img[:,:,1])+(0.0722*img[:,:,2])#perceptually relavent norm
    img = lin2srgb(img)
    
    return img
