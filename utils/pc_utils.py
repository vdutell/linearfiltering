import mkl
mkl.set_num_threads(8)
import numpy as np
from scipy.signal import hilbert
from scipy.signal import hilbert2

import utils.distributions as dists

#Kovesi's phase congruency package
import phasepack as pp


#multithreaded fft support - increases speed for fft & ifft of >100x100 array
#try:
    #notavar #will cause us to use numpy fft
#    import pyfftw
##    import multiprocessing
 #   nthread = multiprocessing.cpu_count()
    #set default functions as faster pyfftw fft
#    def fft2(img):
##        return(pyfftw.interfaces.numpy_fft.fft2(img, threads=nthread))
 #   def ifft2(ft):
#        return(pyfftw.interfaces.numpy_fft.ifft2(ft, threads=nthread))
#except ValueError:
print('Couldn\'t find pyfftw or multiprocessing package. Using default fft')
#set default functions as numpy fft
def fft2(img):
    return(np.fft.fft2(img))
def ifft2(ft):
    return(np.fft.ifft2(ft))


#measure energy of a 1d function
def get_local_energy_1d(f):
    #f = func - np.mean(func)
    h = np.imag(hilbert(f))
    energy = np.sqrt(np.square(f) + np.square(h))
    return(energy) 

def get_local_pc_1d(f):
    e = get_local_energy_1d(f)
    pc = e/np.mean(np.abs(np.fft.fft(f,axis=0)))
    return(pc)

#measure energy over one axis of an image
def measure_energy_axis(img,axis):
    e = np.apply_along_axis(get_local_energy_1d, axis, img) 
    return(e)

##measure pc over one axis of images
def measure_pc_axis(img, axis):
    e = measure_energy_axis(img,axis)
    pc = e/np.mean(np.abs(np.fft.fft2(img,axes=[axis])))
    return(pc)

def measure_energy1_2d(f):
    
    hil = hilbert2(f)
    energy2d = np.sqrt(np.square(np.real(hil))+np.square(np.imag(hil)))
    #phibar = np.arctan(h/f)
    phibar = np.arctan2(np.imag(hil),np.real(hil))
    
    return(energy2d, phibar)


def measure_pc1_2d(img, epsilon=0.001):

    a = np.abs(fft2(img))+epsilon
    e, phibar = measure_energy1_2d(img)
    pc_vals = np.divide(e,np.mean(a))

    return(pc_vals, phibar)

def measure_energy2_2d(f, pbflag=True):

    h = np.imag(hilbert2(f))
    #phibar = np.arctan(h/f)
    phibar = np.arctan2(h,f)
    energy2d = np.multiply((np.cos(phibar) - np.abs(np.sin(phibar))), f) 

    if(pbflag):
        return(energy2d, phibar)
    else:
        return(energy2d)


def measure_pc2_2d(img, epsilon=0.001):

    a = np.abs(fft2(img))+epsilon
    e, phibar = measure_energy2_2d(img)
    pc_vals = np.divide(e,np.sum(a))
    
    return(pc_vals, phibar)


def measure_pc_2d(img, pcn = 1, pbflag=True, epsilon= 0.001):
    # wrapper function to take care of two different pc measurement types
    
    if(pcn==1):
        pc_vals, phibar = measure_pc1_2d(img, epsilon)
    elif(pcn==2):
        pc_vals, phibar = measure_pc2_2d(img, epsilon)
    elif(pcn==3):
        pc_vals = pp.phasecong(img)[0]
        phibar = np.zeros_like(pc_vals) #dont get phibar vals
        
    if(pbflag==False):
        return(pc_vals)
    else:
        return(pc_vals, phibar)

def measure_energy_2d(img, en=1):
    # wrapper function to take care of two different energy measurement types
    if(en==1):
        e_vals, phibar = measure_energy1_2d(img)
    elif(en==2):
        e_vals, phibar = measure_energy2_2d(img)
    return(e_vals, phibar)


def pc_recon_im(phibar, energy):
    #reconstruct an image from phibar and energy values
    im = np.multiply(np.cos(phibar),energy)
    return(im)


def fft_recon_im(amplitude, phase, real_cast=True):
    #reconstruct an image from amplitude and phase values
    complex_recon = ifft2(np.fft.ifftshift(amplitude*np.exp(1j*phase)))
    if(real_cast):
        return(np.real(complex_recon))
    else:
        return(complex_recon)


def gen_pc(image_dims, mean_pc_goal = 0.1, thresh = 0.01, max_iters = 10000, step_size = 0.01, onef_alpha = 1.2, onef_beta = 1.6, wavelet=False, measure='Median'):
    
    '''
    Generate an image patch with 1/f amplitude and a given mean PC value
    
    Parameters:
    image_dims (1x2 float array): Dimensions of image to be generated
    mean_pc_goal (float): Mean PC of ourput image (value to be optimized over)
    thresh (float):  How close to our mean_pc_goal is 'good enough'?
    max_iters:   If we never reach threshold, when should we give up?
    step_size:   What ratio of pi should we step each time?
    onef_alpha:  What is the desired slope of our amplitude of our image?
    onef_beta:     What is the offset of our 1/f slope? (adjusts image contrast)
    wavelet_pc:     Use Kovesi's wavelet package to compute PC?
    
    Returns:
    genimg (2d numpy array): The resulting image patch (may or may not be converged)
    im_seed (2d numpy array): The initial image we started with
    meanpc_evolution (1d numpy array):   The evolution of the PC value over optimization
    '''
    
    #set which pc to calculate
    if(wavelet==True):
        pc_flag = 3
    else:
        pc_flag=1
    
    #loss function 
    def loss_func(amplitude, angle, pc_goal, measure='Median'):
        #trick here use energy for traversing space since it is scalar multiple of PC (and also easier to compute)
        #get complex recon so we can force imaginary component to zero
        real_recon = fft_recon_im(amplitude, angle, real_cast=True)
        #real_recon = np.real(recon_complex)
        pc = measure_pc_2d(real_recon - np.mean(real_recon),
                          pcn=pc_flag)[0]
        if(measure=='Median'):
            mpc = np.median(pc)
        else:
            mpc = np.mean(pc)
        err = np.abs(mpc - pc_goal)
        return(err, real_recon, mpc)
    
    #random 1/f amplitude spectrum
    imamp = dists.make_onef_amp(image_dims, onef_alpha, onef_beta)

    #initialized value for phi (fft phase & value x to be perturbed (dx)).
    init_phi = np.random.rand(*image_dims)*2*np.pi - np.pi
    #make it follow symmetry properties of real signal
    init_phi = dists.rad_symmetrize(init_phi)
    
    phi = init_phi
    im_seed = np.real(np.fft.ifft2(np.fft.ifftshift(imamp*np.exp(1j*init_phi)))) #i
    #im_seed = fft_recon_im(imamp, phi)
    im_seed = im_seed - np.mean(im_seed)
    #resutling error
    loss, recon_complex, pc = loss_func(imamp,phi,mean_pc_goal, measure)
    #count our iterations
    iters = 0
    #evolution of meanpc
    meanpc_evolution = []
    
    #annealing vector
    annealing_vec = np.divide(1.,np.linspace(1,10,num=max_iters))
    #annealing_vec = np.ones(max_iters)
    
    #iterate while our threshold is not reached
    while(loss > thresh):
        #v: random vector we will purturb with
        randphivec = np.random.uniform(-step_size, step_size, size=image_dims)*np.pi
        randphivec = randphivec * annealing_vec[iters]
        tryphi = dists.rad_symmetrize(dists.mod_npi_pi(phi+randphivec))
        tryloss, recon_complex, pc = loss_func(imamp, tryphi, mean_pc_goal, measure)
        if(tryloss < loss):
            phi = tryphi
            loss = tryloss
            
            #calc new PC value to see if we are at threshold
            #real_recon = np.real(recon_complex)
            #pc = measure_pc_2d(real_recon - np.mean(real_recon),
                               #pcn=pc_flag)[0]    

            #keep track of evolution
            meanpc_evolution.append(pc)

        else:
            phi = dists.rad_symmetrize(dists.mod_npi_pi(phi - randphivec))
            #try only updating loss half the time so save computational cost.
            loss, recon_complex, pc = loss_func(imamp, phi, mean_pc_goal, measure)
            meanpc_evolution.append(pc)


        #check iterations
        iters+=1
        if(iters % 1000 == 0):
            print('*',end='')
            #print(pc)
            if(iters %10000 == 0):
                print(iters,end='')
        if(iters >= max_iters):
            break
    
    #check if we reached thresh or needed more iterations
    if(iters >= max_iters):
        print(f'\nDidn\'t get to threshold! stopped at {iters} iterations.')
    else:
        print(f'\nReached Threshold in {iters} iterations!')

    #this is our final image - do a real cast here.
    genimg = np.real(fft_recon_im(imamp, phi))
    genimg = genimg - np.mean(genimg)
    
    # return our final image
    return(genimg, imamp, phi, im_seed, init_phi, meanpc_evolution)

