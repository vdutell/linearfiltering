import torch
import torch.nn as nn
import utils.BNPyramids as bnp
import spyramid as sp
import utils.argmax_diff as amxd
import re
#sys.path.append('../PooledStatisticsMetamers/poolstatmetamer/')

#import poolingregions as pyrpool
#import metamersolver as pyrsolver
import spyramid as spyr

from itertools import combinations_with_replacement

from torchvision.transforms.functional import normalize
#from deeptextures.utils import MEAN, STD

class StatTexGramNet(nn.Module):
    def __init__(self, img_size, batch_size=100, bottleneck_size=100, device=torch.device('cuda:0'), entropy=False,compress=True,color=False,add_marginals=True,add_pspec=False,edge_levels=5,edge_start=1,oris=4,include_dphase=False,boundary_mode='wrap',mag_only=False):
        super(StatTexGramNet, self).__init__()
        
        #self.onehot = onehot #use a hard onehot constraint
        #self.entropy = entropy #use entropy instead of L1 weight loss
        #image params
        self.compress = compress
        self.insize_y, self.insize_x = img_size        
        self.batch_size = batch_size
        self.bottleneck_size = bottleneck_size
        self.device = device
        self.add_marginals = add_marginals
        self.add_pspec = add_pspec
        self.include_dphase = include_dphase
        self.boundary_mode = boundary_mode
        
        self.edge_levels=edge_levels
        self.edge_start=edge_start
        self.oris = oris
        #self.bandpass_levels = self.edge_start

        self.mag_only=mag_only
        
        #self.lowpass_start = edge_start
        #print('lowpass', self.lowpass_start)
        #self.bandpass_start = 0
        #self.bandpass_stop = self.edge_start
        
        self.color = color  
        if(self.color):
            self.color_channels = 3
        else:
            self.color_channels = 1 #color channels
            
        self.dummy_img = torch.zeros(self.batch_size,
                                     self.color_channels,
                                     self.insize_y,
                                     self.insize_x).to(device).contiguous()
        
        params = spyr.SPyramidParams(edge_levels=self.edge_levels,
                                     #edge_start=self.edge_start,
                                     #bandpass_start = 0,
                                     #bandpass_levels = 2,
                                     #lowpass_start=self.lowpass_start,
                                     #bandpass_stop=self.bandpass_stop,
                                     #bandpass_levels=(self.bandpass_start,self.bandpass_stop),
                                     orientations=self.oris,
                                     boundary_mode=self.boundary_mode)
        
        #Cacluate number of pyramids
        if(self.mag_only):
            nedge_pyrs = 1
        else:
            nedge_pyrs = 3 # 1 each for real, imag, and mag
        npyrs = 1 + len(params.lowpass_range()) + oris*nedge_pyrs*len(params.edge_range())
        print(f'There are {npyrs} pyramids per color channel')
        if include_dphase:
            npyrs = npyrs + oris*len(params.crossscale_edge_range()) 
        self.marginal_names = ['mean','var','std']
        self.marginal_size = len(self.marginal_names)
        #self.pspec_names = [f'PS_L{l}' for l in range(self.edge_levels)]
        self.pspec_names = [f'PS_{fq}' for fq in range(img_size[0])]
        self.pspec_size = len(self.pspec_names)

        #BruceNet Encoder Layer
        #if color image
        if(self.color):
            self.bruceNet = bnp.BruceNetPyramidColor(pyramid_params=params, dummy_img=self.dummy_img, include_dphase=self.include_dphase, mag_only=self.mag_only).to(self.device)
            self.npyrs = npyrs*3 #23 before adding imag and real images
            self.gram_size = int((self.npyrs*(self.npyrs+1))//2)
            self.marginal_names = [m+'_C'+c for c in ['1','2','3'] for m in self.marginal_names]
            self.marginal_size = self.marginal_size*3 #one per color channel
            self.pspec_names = [p+'_C'+c for c in ['1','2','3'] for p in self.pspec_names]
            self.pspec_size = self.pspec_size*3 #one per color channel
        else:
            self.bruceNet = bnp.BruceNetPyramid(pyramid_params=params, dummy_img=self.dummy_img,include_dphase=include_dphase).cuda(self.device)
            self.npyrs = npyrs #23 before adding imag and real images
            self.gram_size = int((self.npyrs*(self.npyrs+1))//2)
         
        #Statistics Labels for Gram Matrix
        #ordering of gram matrix combinations
        self.stat_names = self.bruceNet.stat_names
        self.stat_names_gram_label = list(combinations_with_replacement(self.bruceNet.stat_names,r=2))
        self.stat_names_gram_label = [pair[0]+','+pair[1] for pair in self.stat_names_gram_label]
                
        #if including marginals and/or powerspectrum
        self.stats_size = self.gram_size #should be equal to length of stat names gram label
        if(self.add_marginals):
            self.stats_size = self.stats_size + self.marginal_size
            self.stat_names_gram_label = self.stat_names_gram_label + self.marginal_names
        if(self.add_pspec):
            self.stats_size = self.stats_size + self.pspec_size
            self.stat_names_gram_label = self.stat_names_gram_label + self.pspec_names           

        #statisitcs labels
        self.stat_groups_gram_label = self.assign_gram_groups(self.stat_names_gram_label)
        
        #pyramid_encoder
        self.pyr_encoder = nn.Sequential(
            self.bruceNet
        )
        if(self.compress):
            print(f'Compressing from {self.stats_size} to {self.bottleneck_size}')
            self._w = nn.Parameter(torch.rand((self.stats_size, self.bottleneck_size), device=self.device)).contiguous()
            #self._w.requires_grad(True) #set weights to use gradient
        self.triu_indicies = torch.triu(torch.ones((self.npyrs,self.npyrs)),diagonal=0).to(bool)


#     def compressor(x):
#         #self.w = torch.sigmoid(self._w) #habe ich ausgemacht
#         self.w = torch.nn.functional.softmax(self._w,dim=0)
#         x = torch.matmul(x,self.w)
#         #x = torch.relu(x)  # Adding ReLU activation for nonlinearity
#         return(x)-w

    # #gramm matrix calculation
    # def calcgram(self,input_pyramid):
    #     #return(1)
    #     return(torch.mm(input_pyramid.t(),input_pyramid))
    
    def calcgramm(self, tnsr: torch.Tensor, triu_idxes: torch.Tensor) -> torch.Tensor:
        ###FROM pytorch implementation of Leon Gatys Deep Textures
        #https://github.com/trsvchn/deep-textures
        
        """Computes Gram matrix for the input batch tensor.

        Args:
            tnsr (torch.Tensor): input tensor of the Size([B, C, H, W]).

        Returns:
            G (torch.Tensor): output tensor of the Size([B, C, C]).
        """
        N = tnsr.size(1)  # Number of filters (size of feature maps) = C
        M = tnsr.size(-2) * tnsr.size(-1)  # size of the feature maps = H * W
        # Feature map: B, C, H, W ->  B, N, M
        F = tnsr.view(tnsr.size(0), N, M)  # Size([B, N, M])
        # Gram matrix: B, N, M -> B, N, N
        G = F.bmm(F.transpose(1, 2))  # Size([B, N, N])
        G = G[:,triu_idxes]
        return(G)
    
    def calcmarginals(self, rawim: torch.Tensor) -> torch.Tensor:
        '''
        include mean, var,std, skew kurtosis
        '''
        def singlechannel(im):
            mean = im.view(im.size(0), -1).mean(1) #mean but skip batch dim
            mean = mean[:, None, None, None]#expand_dims
            diffs = im - mean
            var = torch.pow(diffs, 2.0)
            var = var.view(var.size(0),-1).mean(1)
            var = var[:, None, None, None] #expand_dims
            std = torch.pow(var, 0.5)
            #zscores = diffs / std
            #skews = torch.pow(zscores, 3.0)
            #skews = skews.view(skews.size(0),-1).mean(1)
            #skews = skews[:, None, None, None] #expand_dims
            #kurtoses = torch.pow(zscores, 4.0)
            #kurtoses = kurtoses.view(kurtoses.size(0),-1).mean(1) - 3.0
            #kurtoses = kurtoses[:, None, None, None] #expand_dims
            #hyperskew = torch.mean(torch.pow(zscores, 5.0))
            #hyperkurt = torch.mean(torch.pow(zscores, 6.0))
            #print(mean.shape,var.shape,std.shape,skews.shape,kurtoses.shape)
            #return(torch.cat([mean,var,std,skews,kurtoses],dim=1).squeeze())
            #print('cat',torch.cat([mean,var,std],dim=1).squeeze(dim=(-2,-1)).shape)
            return(torch.cat([mean,var,std],dim=1).squeeze(dim=(-2,-1)))
            
        if(self.color):
            #ORDER IS: [mean_c1,var_c1,std_c1,mean_c2,var_c2,std_c2,mean_c3,var_c3,std_c3]
            #print('rawim',rawim.shape)
            marginals = torch.cat([singlechannel(rawim.narrow(1,0,1)),
                                   singlechannel(rawim.narrow(1,1,1)),
                                   singlechannel(rawim.narrow(1,2,1))],dim=1)
            #print('marg',marginals.shape)
        else:
            marginals = singlechannel(rawim)
            
        return(marginals)

    #power spectrum - classical
    def calcpowerspec(self, rawim: torch.Tensor) -> torch.Tensor:
        ###MEAN over all orientations at same level of magnitude pyramid, for a given color channel.
        
        """Computes Gram matrix for the input batch tensor.

        Args:
            rawim (torch.tensor): input image

        Returns:
            G (torch.Tensor): output tensor of the Size([B, C, C]).
        """

        def radial_average(power_spectra):
            # Calculate the center of the array
            # Get dimensions and calculate the center of the array
            #print(power_spectra.shape)
            batch_size, _, height, width = power_spectra.shape
            center_y, center_x = height // 2, width // 2
        
            # Create a grid of coordinates centered at the center of the power spectrum
            y, x = torch.meshgrid(torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing='ij')
            r = torch.sqrt((x - center_x)**2 + (y - center_y)**2).to(torch.int64)
            plt.imshow(r)
            plt.show()
        
            # Get the maximum radius
            max_radius = r.max().item()
        
            # Initialize the radial average tensor for the batch
            radial_avgs = torch.zeros(batch_size, max_radius + 1, device=self.device)
        
            # Bin the power spectrum values by their radius across the batch
            for radius in range(max_radius + 1):
                mask = (r == radius)
                mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)  # Expand mask to batch size
                counts = mask.sum(dim=(1, 2))
                
                # Check where we have non-zero counts to avoid division by zero
                valid = counts > 0
                if valid.any():
                    sums = (power_spectra[:,0,:,:] * mask).sum(dim=(1, 2))
                    radial_avgs[valid, radius] = sums[valid] / counts[valid]
        
            return radial_avgs

        def singlechannel(img):
            #sc_power_spec = torch.real(torch.fft.rfft(img).flatten(start_dim=-3))
            #print(img.shape)
            sc_power_spec = torch.real(radial_average(torch.abs(torch.fft.fft2(img))))
            #print(sc_power_spec.shape)
            return(sc_power_spec)

        if(self.color):
            #print(rawim.shape)
            power_spec = torch.cat([singlechannel(rawim.narrow(1,0,1)),
                                   singlechannel(rawim.narrow(1,1,1)),
                                   singlechannel(rawim.narrow(1,2,1))],dim=1)
        else:
            power_spec = singlechannel(rawim)
        return(power_spec)


    #power spectrum from pyramid values.
    def calcpowerspec_pyramid(self, pyrims: torch.Tensor, edge_levels: torch.int, stat_names: list) -> torch.Tensor:
        ###MEAN over all orientations at same level of magnitude pyramid, for a given color channel.
        
        """Computes Gram matrix for the input batch tensor.

        Args:
            pyrims (torch.Tensor): input tensor of the Size([B, C, H, W]).

        Returns:
            power_spec (torch.Tensor): vector of num_levels*3 for color.
        """

        def singlechannel(pi,colorchannel):
            sc_power_spec = torch.zeros((pi.shape[0],edge_levels),device=self.device)
            level_labels = [f'magn_L{l+1}' for l in range(edge_levels)] #levels start at 1
            for i, l in enumerate(level_labels):
                pattern = fr'{re.escape(l)}_O[0-9]_C{colorchannel+1}$'
                level_channels = [i for i, item in enumerate(stat_names) if re.search(pattern, item)]
                level_mean = torch.mean(pi[:,level_channels,:,:])
                sc_power_spec[:,i] = level_mean
            return(sc_power_spec)

        if(self.color):
            # print(singlechannel(pyrims,0))
            # print(singlechannel(pyrims,1))
            # print(singlechannel(pyrims,2))
            power_spec = torch.cat([singlechannel(pyrims,c) for c in range(3)],dim=1)
            #power_spec = torch.cat([singlechannel(pyrims,c) for c in range(3)],dim=1).squeeze(dim=(-2,-1))
        else:
            power_spec = singlechannel(pyrims,0)
        return(power_spec)
    

    def check_same_type(self,label,stattype):
        '''
        Check stat is of a consisten type (orietation, color channel, level, r/i/m) by checking if stat appears twice in the label
        stattype is one of: 'level','ori', 'pyr', 'color'
        '''
    
        if(stattype == 'level'):
            statvals = ['L0','L1','L2','L3','L4','L5','L6','L7','L8','L9'] #assume no more than 10 levels (using 6 right now)
        elif(stattype == 'ori'):
            statvals = ['O0','O1','O2','O3','O4','O5','O6','O7'] #assume no more than 8 orientations (using 4 now)
        elif(stattype == 'color'):
            statvals = ['C1','C2','C3'] #assume no more than 3 color channels
        elif(stattype == 'pyr'):
            statvals = ['real','imag','magn'] #assume no more than 3 color channels
        
        if(any([label.count(statstring)==2 for statstring in statvals])):
            return(True)
        else:
            return(False)

    def assign_gram_groups(self, gramlabels):
        '''
        Start by labeling only stats that are orientated pyramids crosssed with other oriented pyramids that vary in a single dimension (level,orientation,real/imag/mag,color). 
        Name other oriented pyramids that vary in multiple dimensions ori_Xmulti, and anthing crossed with a non-oriented pyramid as highpass,lowpass.
        Marginals are not crossed with anything
    
        TODO: Label cross-scale oriented pyramids as ori_Xscale_d1 if they are neighboring scales.
        TODO: Add power spectrum stats labels
        '''
        
        #labels:
        #'highpass','lowpass','pass_multi','marginal',
        #'ori_Xcolor','ori_Xori','ori_Xlevel','ori_Xri','ori_Xrm','ori_Xim','ori_Xmulti'
        gram_groups = []
        #assign each label to a group
        for gramstatlabel in gramlabels:
            #check for high/low/band pass
            if('highpass' in gramstatlabel):
                #see if it's only highpass, or if it's a highpass cross
                if(gramstatlabel.count('highpass')==2):
                    gram_groups.append('highpass')
                else:
                    gram_groups.append('pass_multi')
            elif('lowpass' in gramstatlabel):
                #see if it's only lowpass, or if it's a lowpass cross
                if(gramstatlabel.count('lowpass')==2):
                    gram_groups.append('lowpass')
                else:
                    gram_groups.append('pass_multi')
            #check marginals *note this will only label marginals x marginals
            elif(any(x in gramstatlabel for x in ['mean','var','std'])):
                gram_groups.append('marginal')

            #check for power spectrum
            elif(any(x in gramstatlabel for x in ['PS_'])):
                gram_groups.append('pspec')
            
            else:
                #now check oriented pyramids.
                #first double check these are only oriented pyaramids 
                assert any(x in gramstatlabel for x in ['real','imag','magn']), f'GramStatLabel {gramstatlabel} should be real, imag, or magn pyramid!'
                #first check for cross color (level, orientation, and r/im/mag must be consistent)
                if(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'pyr')])):
                    gram_groups.append('sub_Xcolor')
                #check for cross orientation (level, color, and r/im/mag must be consistent)
                elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'color'),self.check_same_type(gramstatlabel,'pyr')])):
                    gram_groups.append('sub_Xori')
                #check for cross level (orientation, color, and r/im/mag must be consistent)
                #TODO: Add orientation cross neighbor support (if levels are only one level apart) 'ori_Xlevel_neighbor'
                elif(all([self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color'),self.check_same_type(gramstatlabel,'pyr')])):
                    gram_groups.append('sub_Xlevel')
                #check for cross real/imag pyr (level, orientation, and color must be consistent)
                # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                #     #check for real x imaginary, real x magnitude, and imag x magnitude
                #     if('real' in gramstatlabel and 'imag' in gramstatlabel):
                #         gram_groups.append('sub_Xri')
                #     elif('real' in gramstatlabel and 'magn' in gramstatlabel):
                #         gram_groups.append('sub_Xrm')
                #     elif('imag' in gramstatlabel and 'magn' in gramstatlabel):
                #         gram_groups.append('sub_Xim')

                #check for both real, imag, or mag, and allow variation in one other variable
                #REAL X REAL
                elif(gramstatlabel.count('real')==2):
                    #if(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_real_auto') #this should not be a stat? VD: Yes correct there are zero stats here.
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_real_Xori')
                    # elif(all([self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_real_Xlevel')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori')])):
                    #     gram_groups.append('sub_real_Xcolor')
                    # else:
                    #label these all realx multi for now
                    gram_groups.append('sub_real_Xmulti')
                #IMAG X IMAG                        
                elif(gramstatlabel.count('imag')==2):
                    #label all xmulti for now if(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_imag_auto') #this should not be a stat?
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_imag_Xori')
                    # elif(all([self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_imag_Xlevel')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori')])):
                    #     gram_groups.append('sub_imag_Xcolor')
                    # else:
                    gram_groups.append('sub_imag_Xmulti')
                #MAGN X MAGN
                elif(gramstatlabel.count('magn')==2):
                    # if(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_magn_auto') #this should not be a stat?
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_magn_Xori')
                    # elif(all([self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_magn_Xlevel')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori')])):
                    #     gram_groups.append('sub_magn_Xcolor')
                    gram_groups.append('sub_magn_Xmulti')
                    #gram_groups.append('sub_Xri_Xmulti')
                #if it's cross real/imag/mag, allow variation in one other variable (color/color/level)
                #first cross real/imag
                elif('real' in gramstatlabel and 'imag' in gramstatlabel):
                    if(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                        gram_groups.append('sub_Xri_eq')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xri_Xori')
                    # elif(all([self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xri_Xlevel')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori')])):
                    #     gram_groups.append('sub_Xri_Xcolor')
                    else:
                        gram_groups.append('sub_Xmulti')
                    #gram_groups.append('sub_Xri_Xmulti')
                #next cross real/mag
                elif('real' in gramstatlabel and 'magn' in gramstatlabel):
                    # #print('real','mag: ',gramstatlabel)
                    if(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                        gram_groups.append('sub_Xrm_eq')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xrm_Xori')
                    #     #print('real,mag,cross_ori',gramstatlabel)
                    # elif(all([self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xrm_Xlevel')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori')])):
                    #     gram_groups.append('sub_Xrm_Xcolor')
                    else:
                        gram_groups.append('sub_Xmulti')
                    #gram_groups.append('sub_Xrm_Xmulti')
                #next cross imag/mag
                elif('imag' in gramstatlabel and 'magn' in gramstatlabel):
                    if(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                        gram_groups.append('sub_Xim_eq')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xim_Xori')
                    # elif(all([self.check_same_type(gramstatlabel,'ori'),self.check_same_type(gramstatlabel,'color')])):
                    #     gram_groups.append('sub_Xim_Xlevel')
                    # elif(all([self.check_same_type(gramstatlabel,'level'),self.check_same_type(gramstatlabel,'ori')])):
                    #     gram_groups.append('sub_Xim_Xcolor')
                    else:
                        gram_groups.append('sub_Xmulti')
                        #gram_groups.append('sub_Xim_Xmulti')
                #otherwise, assume we're crossing mulitple stat types
                else:
                    gram_groups.append('garbagecan')
        return(gram_groups)



    #encoder creates pyramid and calculates gramm matrix
    def encoder(self,im):
        #print('pre pyr',im.shape)
        #im = torch.nn.functional.tanh(im)
        im = torch.nn.functional.sigmoid(im)
        xe = self.pyr_encoder(im)
        #print('xe',xe.shape)
        #print('pyramid',x.shape)
        #calculate gram matrix
        x = self.calcgramm(xe,triu_idxes=self.triu_indicies)
        #print('gramm',x.shape)
        if(self.add_marginals):
            x = torch.cat([x,self.calcmarginals(im)],dim=1)
        if(self.add_pspec):
            x = torch.cat([x,self.calcpowerspec(im)],dim=1)
        #if(self.add_pspec_pyr):
        #    x = torch.cat([x,self.calcpowerspec(xe,self.edge_levels,self.stat_names)],dim=1)
        #print('plugmarginals',x.shape)
        return(x)

    #compressor reduces dimensionality (learning)
    def compressor(self,x):
        #self.w = torch.sigmoid(self._w)
        #self.w = torch.nn.functional.softmax(self._w,dim=0)
        self.w = self._w
        x = torch.matmul(x,self.w)
        return(x)
    
    #combine encoder and compressor
    def forward(self,x):
        #print(x.shape)
        x = self.encoder(x)
        if(self.compress):
            x = self.compressor(x)
        return(x)

