import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

import scipy
from scipy.stats import kurtosis
from scipy.stats import entropy
from scipy.stats import moment
import pyrtools as pt

#     return(f_ll)

def calc_phasecon(img, pyramid=True, log=False, collapse=False, order=5, mean_levels=True):
    '''
    Calculate roughness of a texture (based on phase congruency)
    Parameters:
        img (2d Numpy array): Image to have roughness calculated
    Returns:
        f_dir (float): Roughness Coefficient
    '''   
    
    if(pyramid):
        img_pyr = pt.pyramids.SteerablePyramidFreq(img, order=order, is_complex=True)
        pyr_height = int((len(img_pyr.pyr_size)-2)/(order+1))
        pc_vals = []
        
        for pyrlevel in range(pyr_height):
            pc_level_vals = []
            #grab filter image
            vert_filter = img_pyr.pyr_coeffs[(pyrlevel,0)]
            horiz_filter = img_pyr.pyr_coeffs[(pyrlevel,3)]
            diag1_filter = img_pyr.pyr_coeffs[(pyrlevel,1)]
            diag2_filter = img_pyr.pyr_coeffs[(pyrlevel,2)]

            #append magnitude of real and imaginary components to list of vals
            pc_level_vals.append(np.sqrt(np.real(vert_filter)**2 + np.imag(vert_filter)**2))
            pc_level_vals.append(np.sqrt(np.real(horiz_filter)**2 + np.imag(horiz_filter)**2))
            pc_level_vals.append(np.sqrt(np.real(diag1_filter)**2 + np.imag(diag1_filter)**2))
            pc_level_vals.append(np.sqrt(np.real(diag2_filter)**2 + np.imag(diag2_filter)**2))
            
            if(order==5):
                diag3_filter = img_pyr.pyr_coeffs[(pyrlevel,4)]
                diag4_filter = img_pyr.pyr_coeffs[(pyrlevel,5)]
                pc_level_vals.append(np.sqrt(np.real(diag3_filter)**2 + np.imag(diag3_filter)**2))
                pc_level_vals.append(np.sqrt(np.real(diag4_filter)**2 + np.imag(diag4_filter)**2))
            
            if(mean_levels):
                pc_vals.append(np.mean(np.array(pc_level_vals),axis=0))
            else:
                pc_vals.append(np.array(pc_level_vals))

        if(log):
            pc_vals = [np.log10(np.array(p)) for p in pc_vals]
        if(collapse):
            pc_vals = np.mean([np.mean(p) for p in pc_vals])
            
        return(pc_vals)
            
    else:
        #cheap calculation with 2D image (only vertical and horizontal)
        im = img - np.mean(img)
        hil = hilbert2(im)
        energy2d = np.real(np.sqrt(hil**2 + im**2))
        if(log):
            energy2d = np.log10(energy2d) / normalizer
        r = np.mean(energy2d)
    
    return(r)

def calc_phasecon_directional(img,collapse=True):
    order=3
    img_pyr = pt.pyramids.SteerablePyramidFreq(img, order=order, is_complex=True)
    pyr_height = int((len(img_pyr.pyr_size)-2)/(order+1))
    pc_vals_vert = []
    pc_vals_horiz = []
    pc_vals_ldiag = []
    pc_vals_rdiag = []

    for pyrlevel in range(pyr_height):
        #grab filter image
        vert_filter = img_pyr.pyr_coeffs[(pyrlevel,0)]
        horiz_filter = img_pyr.pyr_coeffs[(pyrlevel,3)]
        diag1_filter = img_pyr.pyr_coeffs[(pyrlevel,1)]
        diag2_filter = img_pyr.pyr_coeffs[(pyrlevel,2)]

        #append magnitude of real and imaginary components to list of vals
        pc_vals_vert.append(np.sqrt(np.real(vert_filter)**2 + np.imag(vert_filter)**2))
        pc_vals_horiz.append(np.sqrt(np.real(horiz_filter)**2 + np.imag(horiz_filter)**2))
        pc_vals_ldiag.append(np.sqrt(np.real(diag1_filter)**2 + np.imag(diag1_filter)**2))
        pc_vals_rdiag.append(np.sqrt(np.real(diag2_filter)**2 + np.imag(diag2_filter)**2))
        
    if(collapse):
        pc_vals_vert = np.mean([np.mean(p) for p in pc_vals_vert])
        pc_vals_horiz = np.mean([np.mean(p) for p in pc_vals_horiz])
        pc_vals_ldiag = np.mean([np.mean(p) for p in pc_vals_ldiag])
        pc_vals_rdiag = np.mean([np.mean(p) for p in pc_vals_rdiag])

    pc_vals_dir = [pc_vals_vert, pc_vals_horiz, pc_vals_ldiag, pc_vals_rdiag]
    return(pc_vals_dir)


#set default functions as numpy fft
def fft2(img):
    return(np.fft.fft2(img))
def ifft2(ft):
    return(np.fft.ifft2(ft))


def fft_recon_im(amplitude, phase, real_cast=True):
    
    #reconstruct an image from amplitude and phase values
    complex_recon = ifft2(np.fft.ifftshift(amplitude*np.exp(1j*phase)))
    if(real_cast):
        return(np.real(complex_recon))
    else:
        return(complex_recon)

# def measure_pc_axis(img,axis):
#    

