import numpy as np
import scipy as scp
import pyrtools as pt

def calc_corr_mask(pyrimg1,pyrimg2,max_cor_dist):
    m = max_cor_dist
    #m = max_cor_dist*2+1 #max correlation distance is half of the mask square plus 1
    subband_corr = scp.signal.correlate2d(pyrimg1,pyrimg2)
    #make a mask to take only half of center square
    center_idx = (subband_corr.shape[0]-1)//2 #minus 1 because zero indexed
    corr_mask = np.zeros_like(subband_corr)
    corr_mask[center_idx-(m-1)//2:center_idx+(m-1)//2+1:, #keep center plus half on right side
               center_idx:center_idx+(m-1)//2+1] = 1
    corr_mask[center_idx+1:center_idx+(m+1)//2, #remove duplicates in center column by removing bottom half
               center_idx] = 0
    #keep only the part of the correlation within our mask
    subband_corr = subband_corr.flatten()[corr_maskband.flatten().astype(np.bool)]
    #print(len(subband_corr))
    return(subband_corr)

def get_stats(pyr,im,pyr_height, num_orients):
    
    #### TODO: Make pyramid inside function to save memory
    
    # calculate pyramid height
    #pyr_height = int((len(img_pyr.pyr_size)-2)/(num_orients+1))
    m = 7 #maxiumum autocorrelation distance
    
    ##### create dictionary with statistics #####
    stats_names = []#'pixel_mean','pixel_var','pixel_skew','pixel_kurt','pixel_min','pixel_max', #pixel stats
                   #'pyr_raw_mean_low','pyr_raw_var_low','pyr_raw_min_low','pyr_raw_max_low', #pyramid low pass stats (non oriented)
                   #'pyr_raw_mean_high','pyr_raw_var_high','pyr_raw_min_high','pyr_raw_max_high', #pyramid high pass stats (non oriented)
                   #'pyr_real_autocorrs', #pyramid band stats (oriented)
                   #'pyr_mag_autocorr','pyr_mag_scale_cross_orient','pyr_mag_cross_scale_cross_orient'#pyramid band stats (oriented)
                   
                   #
                   #Central samples of the auto-correlation of each   subband (magnitude) (N K M2+1 parameters),
                   #cross-correlation of each subband 2 (magnitude) with other orientations at the same scale (N  K(K 1) parameters), 
                   #cross-correlation of each subband (magnitude) with other orientations at a coarser scale (K2 (N 1) parameters).
                 # ]
    stats_dict = dict.fromkeys(stats_names,'null')

    ####### fill ditionary with statistics ########
    
    
    ### Marginals ####
    #pixel stats (6)
    raw_img = pyr.recon_pyr().flatten()
    stats_dict['pixel_mean'] = np.mean(raw_img)
    stats_dict['pixel_var'] = np.var(raw_img)
    stats_dict['pixel_skew'] = scp.stats.skew(raw_img.flatten())
    stats_dict['pixel_kurt'] = scp.stats.kurtosis(raw_img.flatten())
    stats_dict['pixel_min'] = np.min(raw_img)
    stats_dict['pixel_max'] = np.max(raw_img)
    
    #High Pass Band Variance (1)
    high_pass_img = pyr.pyr_coeffs['residual_highpass'].flatten()
    stats_dict['pyr_raw_var_high'] = np.var(high_pass_img)
    
    #Skew & Kurtosis of partial reconstruction lowpass image at each scale (2(N+1))
    #lowpass_recon_skews = []
    #lowpass_recon_kurts = []
    for level in range(pyr_height+1):
        partial_recon = pyr.recon_pyr(levels=range(level), bands='all').flatten()
        #lowpass_recon_skews.append(scp.stats.skew(partial_recon))
        #lowpass_recon_kurts.append(scp.stats.kurtosis(partial_recon))
        stats_dict[f'lowpass_recon_skew_L{level}'] = scp.stats.skew(partial_recon)
        stats_dict[f'lowpass_recon_kurt_L{level}'] = scp.stats.kurtosis(partial_recon)
        
    #### Raw Coeffecient Correlations (N+1)*((M^2+1)/2)
    # Autocorr central sample of partially reconstuction lowpass and lowpass band
    # lowpass band autocorr
    low_pass_img = pyr.pyr_coeffs['residual_lowpass']
    recon_lp_residual_autocorr = calc_corr_mask(low_pass_img, low_pass_img, max_cor_dist=m)
    #raster scan into one key per stat
    for i, s in enumerate(recon_lp_residual_autocorr):
        stats_dict[f'autocorr_lowpass_residual_{i}'] = s
    # lowpass reconstruction
    for level in range(pyr_height):
        partial_recon = pyr.recon_pyr(levels=range(level), bands='all')
        recon_autocorr = calc_corr_mask(partial_recon, partial_recon, max_cor_dist=m)
        #raster scan into one key per stat
        for i,s in enumerate(recon_autocorr):
            stats_dict[f'autocorr_lowpass_L{level}_{i}'] = s
            
    ### Coefficient Magnitude Stats ###
    # Autocorrelation of magnitude subbands N*K*((M^2+1)/2)
    for level in range(pyr_height):
        for orient in range(num_orients):
            subband_mag = np.abs(pyr.pyr_coeffs[level,orient])
            subband_autocorr = calc_corr_mask(subband_mag,subband_mag,max_cor_dist=m)
            #raster scan into one key per stat
            for i,s in enumerate(recon_autocorr):
                stats_dict[f'autocorr_magnitude_L{level}_O{orient}_{i}'] = s
    
    # Cross Corr of subband magnitude cross orientation within scale N((k(k-1))/2)
    for level in range(pyr_height):
        for orient1 in range(num_orients-1):
            for orient2 in range(orient1+1,num_orients):
                subband1_mag = np.abs(pyr.pyr_coeffs[level,orient1])
                subband2_mag = np.abs(pyr.pyr_coeffs[level,orient2])
                stats_dict[f'crosscorr_magnitude_L{level}_O{orient1}_O{orient2}'] = np.mean(subband1_mag * subband2_mag)  
                
    # Cross Corr of subband magnitude cross & within orientation at coarser scale K^2(N-1)
    #NOTE: I'm understanding 'at a coarser scale' to mean one single coarser scale based on the number of stats
    for level1 in range(pyr_height-1):
#         for level2 in range(level1+1, pyr_height):
        level2 = level1+1 #one scale coarser
        for orient1 in range(num_orients):
            for orient2 in range(num_orients):
#                 print(level1, level2, orient1, orient2)
                subband1_mag = np.abs(pyr.pyr_coeffs[level1,orient1])
                subband2_mag = np.abs(pyr.pyr_coeffs[level2,orient2])
                subband2_mag = pt.tools.convolutions.upBlur(subband2_mag, n_levels=level2-level1) #Because this is at a higher level need to upscale
                stats_dict[f'coarser_crosscorr_magnitude_L{level1}_L{level2}_O{orient1}_O{orient2}'] = np.mean(subband1_mag * subband2_mag)
                
    ###Cross Scale Phase Stats ###
    # Cross Corr of real with real and imaginary subbands N*K*((M^2+1)/2)
    for level1 in range(pyr_height-1):
        level2 = level1+1 #one scale coarser
        for orient1 in range(num_orients):
            for orient2 in range(num_orients):
                subband1_real = np.real(pyr.pyr_coeffs[level1,orient1])
                #Real x Real
                subband2_real = np.real(pyr.pyr_coeffs[level2,orient2])
                subband2_real = pt.tools.convolutions.upBlur(subband2_real, n_levels=level2-level1) #Because this is at a higher level need to upscale
                stats_dict[f'coarser_crosscorr_real_real_L{level1}_L{level2}_O{orient1}_O{orient2}'] = np.mean(subband1_real * subband2_real)  
                #Real x Imaginary
                subband2_imag = np.imag(pyr.pyr_coeffs[level2,orient2])
                subband2_imag = pt.tools.convolutions.upBlur(subband2_imag, n_levels=level2-level1) #Because this is at a higher level need to upscale
                stats_dict[f'coarser_crosscorr_real_imag_L{level1}_L{level2}_O{orient1}_O{orient2}'] = np.mean(subband1_real * subband2_imag) 
     
    return(stats_dict)


def get_stats_1999(pyr,im,pyr_height, num_orients):
    
    #### TODO: Make pyramid inside function to save memory
    
    ####OLD STATSISTICS LIST FROM P&S 1999 WHICH REPORTS ~1200 STATS. YOU PROBABLY WANT get_stats which returns the 710 stats of P&S 2000, which their matlab code is based off of.####
    # calculate pyramid height
    #pyr_height = int((len(img_pyr.pyr_size)-2)/(num_orients+1))
    m = 7 #maxiumum autocorrelation distance
    
    ##### create dictionary with statistics #####
    stats_names = ['pixel_mean','pixel_var','pixel_skew','pixel_kurt','pixel_min','pixel_max', #pixel stats
                   'pyr_raw_mean_low','pyr_raw_var_low','pyr_raw_min_low','pyr_raw_max_low', #pyramid low pass stats (non oriented)
                   'pyr_raw_mean_high','pyr_raw_var_high','pyr_raw_min_high','pyr_raw_max_high', #pyramid high pass stats (non oriented)
                   'pyr_real_autocorrs', #pyramid band stats (oriented)
                   'pyr_mag_autocorr','pyr_mag_scale_cross_orient','pyr_mag_cross_scale_cross_orient'#pyramid band stats (oriented)
                   
                   #
                   #Central samples of the auto-correlation of each   subband (magnitude) (N K M2+1 parameters),
                   #cross-correlation of each subband 2 (magnitude) with other orientations at the same scale (N  K(K 1) parameters), 
                   #cross-correlation of each subband (magnitude) with other orientations at a coarser scale (K2 (N 1) parameters).
                  ]
    stats_dict = dict.fromkeys(stats_names,'null')

    ####### fill ditionary with statistics ########
    
    #6 pixel stats
    raw_img = pyr.recon_pyr().flatten()
    stats_dict['pixel_mean'] = np.mean(raw_img)
    stats_dict['pixel_var'] = np.var(raw_img)
    stats_dict['pixel_skew'] = scp.stats.skew(raw_img)
    stats_dict['pixel_kurt'] = scp.stats.kurtosis(raw_img)
    stats_dict['pixel_min'] = np.min(raw_img)
    stats_dict['pixel_max'] = np.max(raw_img)
    
    #8 raw pyramid coefficients
    #low pass
    low_pass_img = pyr.pyr_coeffs['residual_lowpass'].flatten()
    stats_dict['pyr_raw_mean_low'] = np.mean(low_pass_img)
    stats_dict['pyr_raw_var_low'] = np.var(low_pass_img)
    stats_dict['pyr_raw_min_low'] = np.min(low_pass_img)
    stats_dict['pyr_raw_max_low'] = np.max(low_pass_img)
    #high pass
    high_pass_img = pyr.pyr_coeffs['residual_highpass'].flatten()
    stats_dict['pyr_raw_mean_high'] = np.mean(high_pass_img)
    stats_dict['pyr_raw_var_high'] = np.var(high_pass_img)
    stats_dict['pyr_raw_min_high'] = np.min(high_pass_img)
    stats_dict['pyr_raw_max_high'] = np.max(high_pass_img)    
    
    #N*K*(M^2+1)/2  Auto Correlations for Real part of each Subband
    autocorrs = []
    for level in range(pyr_height):
        for orient in range(num_orients):
            subband_real = np.real(pyr.pyr_coeffs[level,orient])
            subband_autocorr = calc_corr_mask(subband_real,subband_real,max_cor_dist=m)
            autocorrs.extend(subband_autocorr)
    print(f'{len(autocorrs)} Real Auto Corr Stats')
    stats_dict['pyr_real_autocorrs'] = autocorrs
    
    #N*K*(M^2+1)/2  Auto Correlations for Magnitude of each Subband
    autocorrs = []
    for level in range(pyr_height):
        for orient in range(num_orients):
            subband = pyr.pyr_coeffs[level,orient]
            subband_mag = np.sqrt(np.real(subband)**2 + np.imag(subband)**2)
            subband_autocorr = calc_corr_mask(subband_mag,subband_mag,max_cor_dist=m)
            autocorrs.extend(subband_autocorr)
    print(f'{len(autocorrs)} Magnitude Auto Corr Stats')
    stats_dict['pyr_mag_autocorr'] = autocorrs
    
    #N*K*(M^2+1)/2  Cross Correlation Between Orientation Within Scale for Magnitude of each Subband
    xcorrs = []
    for level in range(pyr_height,0,-1):
        for orient1 in range(num_orients-1):
            for orient2 in range(orient1+1,num_orients):
#                 print(level, orient1, orient2)
                subband1 = pyr.pyr_coeffs[level,orient1]
                subband1_mag = np.sqrt(np.real(subband1)**2 + np.imag(subband1)**2)
                subband2 = pyr.pyr_coeffs[level,orient2]
                subband2_mag = np.sqrt(np.real(subband2)**2 + np.imag(subband2)**2)
                xcorrs.append(np.mean(subband1_mag * subband2_mag))
    print(f'{len(xcorrs)} Cross Orient Cross Scale Stats')
    stats_dict['pyr_mag_scale_cross_orient'] = xcorrs 
    
    #K^2*(N-1)  Cross Correlation Beteween Orientation Between Scale Correlations for Magnitude of each Subband
    xcorrs = []
    for level1 in range(pyr_height-1):
        for level2 in range(level1+1,pyr_height):
#            for orient1 in range(num_orients-1):
#                for orient2 in range(orient1+1,num_orients): #for orientation also want to correlation same orientation at coarser scale.
            for orient1 in range(num_orients):
                for orient2 in range(orient1,num_orients): #for orientation also want to correlation same orientation at coarser scale.
                    #print(level1, level2, orient1, orient2)
                    subband1 = pyr.pyr_coeffs[level1,orient1]
                    subband1_mag = np.sqrt(np.real(subband1)**2 + np.imag(subband1)**2)
                    subband2 = pyr.pyr_coeffs[level2,orient2]
                    subband2_mag = np.sqrt(np.real(subband2)**2 + np.imag(subband2)**2)
                    subband2_mag = pt.tools.convolutions.upBlur(subband2_mag, n_levels=level2-level1) #Because this is at a higher level need to upscale
                    xcorrs.append(np.mean(subband1_mag * subband2_mag))
    print(f'{len(xcorrs)} Cross Orient Cross Scale Stats')
    stats_dict['pyr_mag_cross_scale_cross_orient'] = xcorrs
 
    
    return(stats_dict)
