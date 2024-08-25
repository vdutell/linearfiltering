import numpy as np

def rescale(array):
    #rescale between 0 and 1
    array = array - np.min(array)
    array = array / np.max(array)
    return(array)

def crop_sq(array):
    #crop to square
    mindim = np.min(np.shape(array))
    array = array[:mindim,:mindim]
    return(array)

def crop_matchsize(crop_img,sizematch_img):
    oldy, oldx = crop_img.shape
    newy, newx = sizematch_img.shape
    if((oldy < newy) or (oldx < newx)):
        raise ValueError('Image to be cropped must be larger than sizematched image')
    startx = oldx//2-(newx//2)
    starty = oldy//2-(newy//2)    
    return crop_img[starty:starty+newy,startx:startx+newx]


def rmdc(array):
    #remove DC component
    return(array-np.mean(array))

def make_onef_amp(shape, alpha=1, beta=1.2):
    #make an amplitude spectrum of 1/f
    y, x = np.indices(shape)
    center = np.array(shape)/2
    r = np.hypot(x - center[1], y - center[0])
    amp = np.divide(beta,(1+r**alpha))
    # ensure aplitude spectrum is symmetric.
    amp = rad_symmetrize(amp, odd=False)
    return amp

def make_onef_ims(shape, alpha=1, beta=1.2):
    #make an image with 1/f amplitude and uniformly random phase
    amp_onef = make_onef_amp(shape, alpha, beta) #get amplitude spectrum
    random_phase = np.random.rand(*shape)*2*np.pi - np.pi #get uniformly random phase dist [-pi,pi]
    recon_onef_rand = np.fft.ifft2(np.fft.ifftshift(amp_onef*np.exp(1j*random_phase))) #inverse fft to get image 
    recon_onef_rand = np.real(recon_onef_rand) #real part of image so we can plot and analyze it.
    recon_onef_rand = rescale(recon_onef_rand) #rescale to 0,1
    recon_onef_rand = rmdc(recon_onef_rand)
    return(recon_onef_rand)

def mod_npi_pi(array):
    #convert to [0,2pi]; mod 2pi; convert back to [-pi,pi]
    array = np.add(np.mod(np.add(array,np.pi),2*np.pi),-np.pi)
    return(array)


def rad_symmetrize(full_mat, odd = True):
    '''Make a matrix 'radially odd symmetric' (discarding bottom half)
    Symmetry properties match that of phase spectrum of a real signal
    see: https://ccrma.stanford.edu/~jos/ReviewFourier/Symmetries_Real_Signals.html
    
    Parameters:
    full_mat (2d array):  Matrix we will upper triangular portion of to make resutling herm sym matrix.
    odd (bool): Is bottom of matrix negaive of top? (Set this to True for phase spectra, and false for amplitude spectra)
    
    Returns:
    herm_mat (2d array):  Matrix based on full_mat that is Hermetian Symmetric
    
    '''
    
    #check for even number of rows & pull of top row if needed
    if((full_mat.shape[0] % 2) == 0):
        add_bonus_row = True
        bonus_row, full_mat = np.vsplit(full_mat, [1,])
    else:
        add_bonus_row = False
    #check for even number of cols & pull off top col if needed
    if((full_mat.shape[1] % 2) == 0):
        add_bonus_col = True
        bonus_col, full_mat = np.hsplit(full_mat, [1,])
    else:
        add_bonus_col = False
    
    mat_top = full_mat[:np.shape(full_mat)[0]//2]
    #make bottom hermmetian symmetric wrt top
    mat_bottom  = np.flipud(np.fliplr(mat_top))
    if(odd):
        mat_bottom = -1 * mat_bottom

    #make mat middle horizontally symmetric
    mat_middle = full_mat[np.shape(full_mat)[0]//2]
    mm = mat_middle[:np.shape(mat_middle)[0]//2][::-1]
    if(odd):
        mm = -1*mm
    mat_middle[np.shape(mat_middle)[0]//2+1 :] = mm
    #remove DC component
    mat_middle [np.shape(mat_middle)[0]//2] = 0
    
    #put our matrix back together
    new_mat = np.vstack((mat_top, mat_middle, mat_bottom))
    
    #add back our extra columns and rows if needed
    if(add_bonus_col):
        new_mat = np.hstack((bonus_col,new_mat))
    if(add_bonus_row):
        new_mat = np.vstack((bonus_row,new_mat))
    
    return(new_mat)
