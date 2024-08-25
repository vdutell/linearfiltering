import torch
import torch.nn as nn

import utils.encoder_stat_sets as sests
import imp
imp.reload(sests)

import sys
sys.path.append('../pmetamer')  # Hack to allow importing from pmetamer sibling package/directory
import spyramid as spyramid

class StatNetEncoder(nn.Module):
    def __init__(self, img_size, batch_size, num_stats, vectorized, convolutional, manual_stats, device):
        super(StatNetEncoder, self).__init__()
        
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_stats = num_stats
        self.vectorized = vectorized
        self.convolutional = convolutional
        self.manual_stats = manual_stats
        self.device = device
        
        #input sizes - constants for now and the forseeable future
        self.color_channels = 1 #color channels
        self.dummy_img = torch.zeros(self.batch_size,
                                     self.color_channels,
                                     self.img_size[0],
                                     self.img_size[1]).to(device)
        self.spatial_kernel_size = 8
        self.num_levels = 5
        self.num_bands = 4
        self.num_pyr_ims = (self.num_levels-2) * self.num_bands * 2 + 2
        
        #steerable pyramid data structure
        #self.steerable_pyr = SCFpyr_PyTorch(height=self.num_levels, 
        #                                    nbands=self.num_bands, 
        #                                    scale_factor=2, 
        #                                    device=self.device)
        #pyr_coeffs = self.steerable_pyr.build(self.dummy_img)
        
        #pmetamer pyramid strucutre (& Stats!)
        self.steerable_pyr = pemtamer.

        #vectorized version of network - fully connected network
        if(self.vectorized):
            #calculate size of linear 
            total = pyr_coeffs[0].flatten().shape[0]+pyr_coeffs[-1].flatten().shape[0]
            for i in range(1,len(pyr_coeffs)-1):
                for o in pyr_coeffs[i]:
                    total += o.flatten().shape[0]
            self.vector_size = int(total / self.batch_size)

            #neural network encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.vector_size, self.num_stats),
                nn.Sigmoid()
            )
        #spatially localized version of network
        elif(self.convolutional):            
            #neural network encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels = self.num_pyr_ims, 
                          out_channels = self.num_stats,
                          kernel_size = (self.spatial_kernel_size,
                                         self.spatial_kernel_size),
                          stride = 1),
                ##Start here. 5D convoltuion is expensive (and not implemented)
                ## Consider both spatial and channel kernels, or  the following family of kernels:
                    #k1 crossspatial kernels size nxnx1x1x1 3D convolved all but spatial dime
                    #k2 crosscomplex kernels size 1x1x2x1x1 4D convolved all but real/complex dim
                    #k3 crossorient kernels size 1x1x1x4x1 4D convolved all but orientation dim
                    #k4 crossorient kernels size 1x1x1x1x3 4D convolve  all but scale dim
                nn.Sigmoid()
            )
                
        elif(self.manual_stats):
            self.num_stats_fullset = 5000
            #neural network encoder
            self.encoder = nn.Sequential(
                #nn.utils.weight_norm(nn.Linear(self.num_stats_fullset, self.num_stats),dim=0)
                nn.Linear(self.num_stats_fullset, self.num_stats)
            )
        else:
            print('No Encoder Specified! Must used vectorized, convolutional, or manualstats!')
            

    def encode_vectorized(self, x):
        #print('x',x.type)
        pyr_coeffs = self.steerable_pyr.build(x)
        pyr_coeffs = utils.vectorize_batch(pyr_coeffs)
        #print(pyr_coeffs.shape)
        x = self.encoder(pyr_coeffs)
        #print(x.shape)
        return(x)
    
    def encode_upsampled(self, x):
        #print('x',x.type)
        pyr_coeffs = self.steerable_pyr.build(x)
        pyr_coeffs = utils.upsample_batch(pyr_coeffs)
        x = self.encoder(pyr_coeffs)
        #print(x.shape)
        return(x)
    
    def encode_manualstats(self, img):
        #print('x',x.type)
        x = self.steerable_pyr.build(img)
        x = utils.upsample_batch(x)
        x = sests.calc_manualstats(x,img)
        x = self.encoder(x)
        #print(x.shape)
        return(x)
    
    def encode(self,x):
        if self.vectorized:
            return(self.encode_vectorized(x))
        elif self.convolutional:
            return(self.encode_upsampled(x))
        elif self.manual_stats:
            return(self.encode_manualstats(x))
        else:
            print('ERRORRRRRR in encoder')
