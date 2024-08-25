import sys
#sys.path.append('../pmetamer/pmetamer/')
sys.path.append('../PooledStatisticsMetamers/poolstatmetamer/')

import poolingregions as pyrpool
import metamersolver as pyrsolver
import torch
import torch.nn as nn

class BruceNet(nn.Module):
    'Forward Pass of Bruces Metamer Generator Network'
    def __init__(self, pooling_region_size, 
                 pyramid_params, dummy_img):
        super().__init__()
        self.pool = pyrpool.make_uniform_pooling(pooling_region_size) 
        self.solver = pyrsolver.make_solver(target_image=dummy_img,
                                            stat_pooling=self.pool, 
                                        pyramid_params=pyramid_params)
        
    #def getstats(self):
    #    statset = self.mimg.get_statgroups()
    #    return(statset)
    
    def setstats(self, statnamelist, boolflaglist):
        for stat,flag in zip(statnamelist,boolflaglist):
            self.solver.set_stat_mode(stat,flag)
        print('Statistics Set in Solver by BruceNet')
        
    def calcstats(self, current_frame, get_labels=False):
        stats = self.solver.stat_eval(current_frame, create_labels=get_labels)
        if(get_labels):
            labels = self.solver.stat_eval.get_all_labels()
            return(labels)
        else:
            #print('len',len(stats))
            #print([s.shape for s in stats])
            #will need to change this for color, dim is removed as is
            stats = torch.stack([s.squeeze() for s in stats])
            #print(stats[2])
            #print('stsh',stats.shape)#.squeeze(2) #brucenet passes a list of size numstats
            stats = torch.transpose(stats,1,0) #transpose batch and list dims
            #print('stats shape:',stats.shape)
            #print(len(stats),stats[1])
            return(stats)
        
    def forward(self, x):
        stats = self.calcstats(x)
        return(stats)
