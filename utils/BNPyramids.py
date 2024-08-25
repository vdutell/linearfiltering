import sys
#sys.path.append('../pmetamer/pmetamer/')
sys.path.append('../PooledStatisticsMetamers/poolstatmetamer/')

#import poolingregions as pyrpool
#import metamersolver as pyrsolver
import spyramid as spyr
import color_utils as color

import torch
import torch.nn as nn

class BruceNetPyramid(nn.Module):
    'Pyramid Portion of Forward Pass of Bruces Metamer Generator Network'
    def __init__(self, pyramid_params, dummy_img,include_dphase=False):
        super().__init__()
        #self.params = SPyramidParams(4,boundary_mode='wrap')
        self.filter_names = ['low_pass_image',
                             'edge_real_images',
                             'edge_imag_images',
                             'edge_magnitude_images',
                             'high_pass_image']
        # if pyramid_params==None:
        #     self.params = spyr.SPyramidParams(4,boundary_mode='wrap')
        # else:
        #     
        self.params = pyramid_params
        self.builder = spyr.SPyramidFourierBuilder(dummy_img,self.params,downsample=True)
        
        self.include_dphase = include_dphase
        print('Dphase: ',include_dphase)
        # self.pool = pyrpool.make_uniform_pooling(pooling_region_size) 
        # self.solver = pyrsolver.make_solver(target_image=dummy_img,
        #                                     stat_pooling=self.pool, 
        #                                 pyramid_params=pyramid_params)
        
        #HARDCODED!!! Should update this to be variable with the functions in vectorize_pyramid
        #NOTE: LEVEL 1 is highest frequency for edge magnitude filters.
        #VD 12/6/23 validated that orientation loops inside scale in vectorize pyramid.
        #VD: 1/29/24 Need to include dphase image names if using
        self.stat_names = [f'lowpass_{i}' for i in self.params.lowpass_range()] + \
                            [f'real_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \
                            [f'imag_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \
                            [f'magn_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \
                            [f'highpass']
#                            #[f'real_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \
    
    def vectorize_pyramid(self, spyramid):
        '''
        go through images and vectorize them all
        '''
        #create a stack treating pyramid images as channels
#         high_pass = spyramid.high_pass_image()
#         band_passes = torch.cat([spyramid.band_pass_image(i) for i in spyramid.params().bandpass_range()],dim=1)
#         low_pass = torch.cat([spyramid.low_pass_image(i) for i in spyramid.params().lowpass_range()],dim=1)
#         edge_real = torch.cat([spyramid.edge_real_images(i) for i in spyramid.params().edge_range()],dim=1)
#         edge_imag = torch.cat([spyramid.edge_imag_images(i) for i in spyramid.params().edge_range()],dim=1)
#         edge_mags = torch.cat([spyramid.edge_magnitude_images(i) for i in spyramid.params().edge_range()],dim=1)
#         return(torch.cat([low_pass,edge_real,edge_imag,edge_mags,band_passes,high_pass],dim=1))
        
        return(torch.cat([
            spyramid.high_pass_image(),
            torch.cat([spyramid.low_pass_image(i) for i in spyramid.params().lowpass_range()],dim=1),
            torch.cat([spyramid.edge_real_images(i) for i in spyramid.params().edge_range()],dim=1),
            torch.cat([spyramid.edge_imag_images(i) for i in spyramid.params().edge_range()],dim=1),
            torch.cat([spyramid.edge_magnitude_images(i) for i in spyramid.params().edge_range()],dim=1)
            ]))


    def forward(self, x):
        x = self.builder.build_spyramid(x)
        x = self.vectorize_pyramid(x)
        return(x)


class BruceNetPyramidColor(nn.Module):
    '''
    Wrapper for BruceNetPyramid to support color images
    '''
    def __init__(self,pyramid_params,dummy_img,include_dphase=False,mag_only=False):
        super().__init__()
        
        self.colorspace = color.RGBToOpponentConeTransform()
        self.channel_names = self.colorspace.channel_names()
        self.include_dphase = include_dphase
        self.mag_only = mag_only
        

        # if pyramid_params==None:
        #     self.params = spyr.SPyramidParams(,boundary_mode='wrap')
        # else:
        #     
        self.params = pyramid_params
        
        #leave out cross color stats for now - individual channels only
        #we will get cross color stats from gramm matrix calculation downstream
        #downsample must be false here because need to stack all pyramids together for gramm
        self.builder_1 = spyr.SPyramidFourierBuilder(dummy_img[0],self.params,downsample=False)
        self.builder_2 = spyr.SPyramidFourierBuilder(dummy_img[0],self.params,downsample=False)
        self.builder_3 = spyr.SPyramidFourierBuilder(dummy_img[0],self.params,downsample=False)
        #self.xcolorstats = 
        
        #ordering of gram matrix combinations
        #HARDCODED!!! Should update this to be variable with the functions in vectorize_pyramid
        #NOTE: LEVEL 1 is highest frequency for edge magnitude filters.
        #VD 12/6/23 validated that orientation loops inside scale in vectorize pyramid.
        #VD: 1/29/24 Need to include dphase image names if using
        #VD 2/19 Included support for Magnitude only but this does not include dphase image support.
        if(self.mag_only):
            self.single_channel_stat_names = [f'lowpass_{i}' for i in self.params.lowpass_range()] + \
                                [f'magn_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \
            [f'highpass']            
        else:
            self.single_channel_stat_names = [f'lowpass_{i}' for i in self.params.lowpass_range()] + \
                                [f'real_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \
                                [f'imag_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \
                                [f'magn_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \
                            [f'highpass']

# [f'real_L{i}_O{l}' for i in self.params.edge_range() for l in range(self.params.orientations)] + \

        self.stat_names = [s+'_C1' for s in self.single_channel_stat_names] + \
                           [s+'_C2' for s in self.single_channel_stat_names] + \
                           [s+'_C3' for s in self.single_channel_stat_names]
        
    def vectorize_pyramid(self, spyramid):
        '''
        go through images and vectorize them all
        '''
        #create a stack treating pyramid images as channels
        high_pass = spyramid.high_pass_image()
        #'BAND PASS IMAGES' ARE A SINGLE IMAGE THAT IS IDENTICAL TO THE HIGH PASS IMAGE SKIP IT!
        #band_passes = torch.cat([spyramid.band_pass_image(i) for i in spyramid.params().bandpass_range()],dim=1)
        low_pass = torch.cat([spyramid.low_pass_image(i) for i in spyramid.params().lowpass_range()],dim=1)
        edge_mags = torch.cat([spyramid.edge_magnitude_images(i) for i in spyramid.params().edge_range()],dim=1)
        #debugging shapes
        # print('individual shapes')
        # print(high_pass.shape,band_passes.shape, low_pass.shape, edge_real.shape,edge_imag.shape,edge_mags.shape)
        # print('fullshape')
        # print(torch.cat([low_pass,edge_real,edge_imag,edge_mags,band_passes,high_pass],dim=1).shape)
        # print(spyramid.params().edge_range())
        # print('number pyrs')
        # print(len(spyramid.params().bandpass_range()),
        #       len(spyramid.params().lowpass_range()),
        #       len(spyramid.params().edge_range()))
        # npyrs = 1 + 1 + len(spyramid.params().bandpass_range())+1 + len(spyramid.params().lowpass_range())+1 + 3*len(spyramid.params().edge_range())+1

        #note magnitude only option will not return dphase images (OR LABELS)
        if self.mag_only:
            return(torch.cat([low_pass,edge_mags,high_pass],dim=1))
        else:
            edge_real = torch.cat([spyramid.edge_real_images(i) for i in spyramid.params().edge_range()],dim=1)
            edge_imag = torch.cat([spyramid.edge_imag_images(i) for i in spyramid.params().edge_range()],dim=1)
            
        if(self.include_dphase==False):
            return(torch.cat([low_pass,edge_real,edge_imag,edge_mags,high_pass],dim=1))
            #return(torch.cat([low_pass,edge_real,edge_mags,high_pass],dim=1))
            #TESTING REMOVING EDGE Imaginary
            #return(torch.cat([low_pass,edge_mags,edge_real,band_passes,high_pass],dim=1))
        # else:
        #     #print('returning dphase')
        #     #dphase_real = torch.cat([spyramid.dphase_real_images(i) for i in spyramid.params().edge_range()],dim=1)
        #     #dphase_imag = torch.cat([spyramid.dphase_imag_images(i) for i in spyramid.params().crossscale_edge_range()],dim=1)
        #     #print(spyramid.params().crossscale_edge_range())
        #     #print(dphase_imag.shape)
        #     #return(torch.cat([low_pass,edge_real,edge_imag,edge_mags,high_pass,dphase_imag],dim=1))
        #     #return(torch.cat([low_pass,edge_real,edge_mags,high_pass,dphase_imag],dim=1))
        #     #return(torch.cat([low_pass,edge_imag,edge_mags,high_pass,dphase_imag],dim=1))

        
    def forward(self,x):
        #print(x.is_cuda)
        x = self.colorspace(x)
       # print(x.narrow(1,0,1).is_cuda)
       # print(self.builder_1.is_cuda)
        #print(self.builder_2.build_spyramid(x.narrow(1,0,1)).is_cuda)
        # test = self.builder_1.build_spyramid(x.narrow(1,0,1))
        #print('narrow',x.narrow(1,0,1).shape)
        x = torch.cat([self.vectorize_pyramid(self.builder_1.build_spyramid(x.narrow(1,0,1),colorname=self.channel_names[0])),
                       self.vectorize_pyramid(self.builder_2.build_spyramid(x.narrow(1,1,1),colorname=self.channel_names[1])),
                       self.vectorize_pyramid(self.builder_3.build_spyramid(x.narrow(1,2,1),colorname=self.channel_names[2]))],
                       dim=1)
        # print(x.shape)
        # x = torch.cat([self.vectorize_pyramid(x[0]),
        #                self.vectorize_pyramid(x[1]),
        #                self.vectorize_pyramid(x[2])])
        return(x)
        #ignore xchannel stats for now (they are in metaermstatevel line 81)

        
