import torch
import torch.nn as nn
import steerable.utils as utils
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch

def calc_manualstats(pyramid_tensor, img):
    
    #cross-band marginals
    statstensor_names = ['crossband_mean',
                         'crossband_var',
                         'crossband_skew',
                         'crossband_kurt',
                         'crossband_min',
                         'crossband_max',
                         ]
    print(img.shape)
    print(torch.min(img.view(),dim=[1,2,3]).shape)
    #print(torch.mean(img,dim=[1,2,3]).shape)
    #print(torch.std(img,dim=[1,2,3]).shape)
    #print(img.flatten(start_dim=1).shape)
    #mean_broadcasted = torch.mean(img,dim=[1,2,3]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #std_broadcasted = torch.std(img,dim=[1,2,3]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    print(mean_broadcasted.shape)
    mean = torch.mean(img,dim=[1,2,3],keepdim=True)
    std = torch.std(img,dim=[1,2,3],keepdim=True)
    print(mean.shape)
    print(std.shape)
    statstensor = torch.cat([mean.squeeze(),
                            torch.var(img,dim=[1,2,3]),
                            #torch.mean((img-mean)/std)**3).squeeze(),
                            #torch.mean((img-mean)/std)**4).squeeze(),
                            torch.min(img,dim=[1,2,3],keepdim=True),
                            torch.max(img,dim=[1,2,3],keepdim=True)],
                            dim=-1
                            )
    print(f'statstensorsize:, {statstensor.shape}')
    #within-band marginals
    statstensor_names = ['withinband_mean',
                         'withinband_var',
                         'withinband_skew',
                         'withinband_kurt',
                         'withinband_min',
                         'withinband_max',
                         ]
    statstensor = torch.cat([statstensor,
                            torch.mean(img,dim=[2,3]),
                            torch.var(img,dim=[2,3]),
                            torch.mean(((img-torch.mean(img,dim=[2,3]))\
                                        /torch.std(img,dim=[2,3]))**3,dim=[2,3]),
                            torch.mean(((img-torch.mean(img,dim=[2,3]))\
                                        /torch.std(img,dim=[2,3]))**4,dim=[2,3]),
                            torch.min(img,dim=[2,3]),
                            torch.max(img,dim=[2,3])],
                            dim=1
                           )
    print(f'statstensorsize:, {statstensor.shape}')

    #SKIP SKEW & KURT OF PARTIAL RECON LOWPASS AT EACH SCALE FOR NOW
    #ALSO SKIP CENTRAL SAMPLE OF AUTOCORR OF PARTIALLY RECONSTRUCTED LOWPASS IMG AND LOWPASS BAND
    
    #coeffieint magnitude
    statstensor = torch.cat([statstensor,
                            torch.tensor([acor.autocorrelation_image(i, offset, offset_scale=s, center=True) for i in pyramid_tensor for s in offset_scale])],
                            dim=1)
    print(f'statstensorsize:, {statstensor.shape}')
    return(statstensor)
