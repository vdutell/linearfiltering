import matplotlib.pyplot as plt
import utils.pc_utils as pcu
import utils.distributions as dists

import numpy as np

plt.rcParams['figure.figsize'] = [16, 16]

def pc_evolution(meanpc_evolution, runinfo_string, show=True, measure='Median'): 
    plt.figure(figsize=(10,10))
    plt.plot(meanpc_evolution)
    plt.title(f'Evolution of {measure} PC')
    plt.savefig(f'output/evolution_'+ runinfo_string +'.png',dpi=300)
    if(show):
        plt.show()
    else:
        plt.close()

def compare_initim_genim(genim, initim, runinfo_string, show=True):
    
    #plot init im
    plt.subplot(121)
    plt.imshow(initim, cmap='Greys_r')
    plt.axis('off')
    plt.title(f'Initial: {runinfo_string}')
    plt.subplot(122)
    #plot generated im
    plt.imshow(genim,cmap='Greys_r')
    plt.axis('off')
    plt.title(f'Generated: {runinfo_string}')
    plt.savefig(f'output/initial_generated_ims_'+ runinfo_string +'.png',dpi=300)
    if(show):
        plt.show()
    else:
        plt.close()

def compare_initim_genim_stats(genim, amp, ggp, initim, igp, runinfo_string, show=True):
    plt.figure(figsize=(10,4))
    ipc, ipb = pcu.measure_pc_2d(initim)
    #ift = np.fft.fftshift(np.fft.fft2(initim))
    #iap = np.abs(ift)
    #igp = np.angle(ift)
    plt.subplot(251)
    plt.imshow(initim,cmap='Greys_r')
    plt.axis('off')
    plt.title('Init Im')
    plt.subplot(252)
    plt.imshow(ipc,cmap='Greys_r')
    plt.axis('off')
    plt.title('Init PC')
    plt.subplot(253)
    plt.imshow(ipb,cmap='hsv')
    plt.title('Init PhiB')
    plt.axis('off')
    plt.subplot(254)
    plt.imshow(amp,cmap='Greys_r')
    plt.title('Init Amp')
    plt.axis('off')
    plt.subplot(255)
    plt.imshow(igp,cmap='hsv')
    plt.title('Init GP')
    plt.axis('off')

    gpc, gpb = pcu.measure_pc_2d(genim)
    gft = np.fft.fftshift(np.fft.fft2(genim))
    gap = np.abs(gft)
    ggp = np.angle(gft)
    plt.subplot(256)
    plt.imshow(genim,cmap='Greys_r')
    plt.title('Gen Im')
    plt.axis('off')
    plt.subplot(257)
    plt.title('Gen PC')
    plt.imshow(gpc,cmap='Greys_r')
    plt.axis('off')
    plt.subplot(258)
    plt.title('Gen PhiB')
    plt.imshow(gpb,cmap='hsv')
    plt.axis('off')
    plt.subplot(259)
    plt.title('Gen Amp')
    plt.imshow(amp,cmap='Greys_r')
    plt.axis('off')
    plt.subplot(2,5,10)
    plt.title('Gen GP')
    plt.imshow(ggp,cmap='hsv')
    plt.axis('off')
    plt.savefig(f'output/initial_generated_pcpb_'+ runinfo_string +'.png', 
                    dpi=800)
    if(show):
        plt.show()
    else:
        plt.close()

def hist_pc_dists(genim, initim, cats, trail, beach, runinfo_string, show=True,save=True):
    plt.figure(figsize=(10,10))
    aval=0.2
    
    #get colors
    cs = plt.rcParams['axes.prop_cycle'].by_key()["color"]
    
    # generated image
    pc = pcu.measure_pc_2d(genim)[0]
    plt.hist(pc.flatten(),
             bins=50,density=True,label='generated', alpha = aval);
    plt.axvline(np.median(pc), c=cs[0])
    # initial image
    pc = pcu.measure_pc_2d(initim)[0]
    plt.hist(pc.flatten(),
             bins=50,density=True,label='initial', alpha = aval);
    plt.axvline(np.median(pc), c=cs[1])
    # cats
    pc = pcu.measure_pc_2d(dists.crop_matchsize(cats,genim))[0]
    plt.hist(pc.flatten(),
             bins=50,density=True,label='cats', alpha = aval);
    plt.axvline(np.median(pc), c=cs[2])
    # trail
    #pc = pcu.measure_pc_2d(dists.crop_matchsize(trail,genim))[0]
    #p = plt.hist(pc.flatten(),
    #         bins=50,density=True,label='trail', alpha = aval);
    #plt.axvline(np.median(pc), c=p[0].getcolor())
    # beach
    pc = pcu.measure_pc_2d(dists.crop_matchsize(beach,genim))[0]
    plt.hist(pc.flatten(),
             bins=50,density=True,label='beach',alpha = aval);
    plt.axvline(np.median(pc), c=cs[3])
    
    plt.title('Distribution of Pixelwise PC')
    plt.legend()

    if(save):
        plt.savefig(f'output/initial_generated_dists_'+ runinfo_string +'.png', dpi=300)
    if(show):
        plt.show()
    else:
        plt.close()
