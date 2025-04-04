import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones import Conv_4,ResNet


class invertedBlock(nn.Module):
    def __init__(self, in_channel, out_channel,ratio=2):
        super(invertedBlock, self).__init__()
        internal_channel = in_channel * ratio
        self.relu = nn.GELU()
        ## 7*7卷积，并行3*3卷积
        self.conv1 = nn.Conv2d(internal_channel, internal_channel, 7, 1, 3, groups=in_channel,bias=False)

        self.convFFN = ConvFFN(in_channels=in_channel, out_channels=in_channel)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.pw1 = nn.Conv2d(in_channels=in_channel, out_channels=internal_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channel, out_channels=in_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)

    def hifi(self, x):
        x1 = self.pw1(x)
        x1 = self.relu(x1)
        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        x1 = self.pw2(x1)
        x1 = self.relu(x1)
        # x2 = self.conv2(x)
        x3 = x1+x

        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.layer_norm(x3)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x4 = self.convFFN(x3)

        return x4

    def forward(self, x):
        return self.hifi(x)+x


class ConvFFN(nn.Module):

    def __init__(self, in_channels, out_channels, expend_ratio=4):
        super().__init__()

        internal_channels = in_channels * expend_ratio
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        x1 = self.pw1(x)
        x2 = self.nonlinear(x1)
        x3 = self.pw2(x2)
        x4 = self.nonlinear(x3)
        return x4 + x

class MSAG(nn.Module):
    """
    Multi-scale attention gate
    Arxiv: https://arxiv.org/abs/2210.13012
    """
    def __init__(self, channel):
        super(MSAG, self).__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 3, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)
        _x = self.relu(torch.cat((x1, x2, x3), dim=1))
        _x = self.voteConv(_x)
        x = x + x * _x
        return x

class FRN(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False,is_pretraining=False,num_cat=None):
        
        super().__init__()

        if resnet:
            num_channel = 640
            self.feature_extractor = ResNet.resnet12()

        else:
            num_channel = 64
            self.feature_extractor = Conv_4.BackBone_final(num_channel)

        self.shots = shots
        self.way = way
        self.resnet = resnet

        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel
        self.msag = MSAG(num_channel)
        self.rep1 = nn.Sequential(*[invertedBlock(in_channel=num_channel, out_channel=num_channel,ratio=1) for i in range(1)])
        
        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)
        
        # H*W=5*5=25, resolution of feature map, correspond to r in the paper
        self.resolution = 25 

        # correpond to [alpha, beta] in the paper
        # if is during pre-training, we fix them to 0
        self.r = nn.Parameter(torch.zeros(2),requires_grad=not is_pretraining)

        if is_pretraining:
            # number of categories during pre-training
            self.num_cat = num_cat
            # category matrix, correspond to matrix M of section 3.6 in the paper
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat,self.resolution,self.d),requires_grad=True)   

    def get_feature_map(self,inp):

        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)
        feature_map_new = self.msag(feature_map)
        feature_map_final = self.rep1(feature_map_new)
        #feature_map_final = torch.cat((feature_map_new, feature_map_final), dim=1)

        if self.resnet:
            feature_map_final = feature_map_final/np.sqrt(640)
        
        return feature_map_final.view(batch_size,self.d,-1).permute(0, 2, 1).contiguous()  # N,HW,C

    def get_recon_dist(self,query,support,alpha,beta,Woodbury=True):
    # query: way*query_shot*resolution, d
    # support: way, shot*resolution , d
    # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(1)/support.size(2)
        
        # correspond to lambda in the paper
        lam = reg*alpha.exp()+1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0,2,1) # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper
            
            sts = st.matmul(support) # way, d, d
            m_inv = (sts+torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse() # way, d, d
            hat = m_inv.matmul(sts) # way, d, d
        
        else:
            # correspond to Equation 8 in the paper
            
            sst = support.matmul(st) # way, shot*resolution, shot*resolution
            m_inv = (sst+torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() # way, shot*resolution, shot*resolutionsf 
            hat = st.matmul(m_inv).matmul(support) # way, d, d

        Q_bar = query.matmul(hat).mul(rho) # way, way*query_shot*resolution, d

        dist = (Q_bar-query.unsqueeze(0)).pow(2).sum(2).permute(1,0) # way*query_shot*resolution, way
        
        return dist

    def get_neg_l2_dist(self,inp,way,shot,query_shot,return_support=False):
        
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]

        feature_map = self.get_feature_map(inp)

        support = feature_map[:way*shot].view(way, shot*resolution , d)
        query = feature_map[way*shot:].view(way*query_shot*resolution, d)

        recon_dist = self.get_recon_dist(query=query,support=support,alpha=alpha,beta=beta) # way*query_shot*resolution, way
        neg_l2_dist = recon_dist.neg().view(way*query_shot,resolution,way).mean(1) # way*query_shot, way
        
        if return_support:
            return neg_l2_dist, support
        else:
            return neg_l2_dist

    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)

        _,max_index = torch.max(neg_l2_dist,1)

        return max_index

    def forward_pretrain(self,inp):

        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)

        feature_map = feature_map.view(batch_size*self.resolution,self.d)
        
        alpha = self.r[0]
        beta = self.r[1]
        
        recon_dist = self.get_recon_dist(query=feature_map,support=self.cat_mat,alpha=alpha,beta=beta) # way*query_shot*resolution, way

        neg_l2_dist = recon_dist.neg().view(batch_size,self.resolution,self.num_cat).mean(1) # batch_size,num_cat
        
        logits = neg_l2_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction

    def forward(self,inp):

        neg_l2_dist, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)
            
        logits = neg_l2_dist*self.scale
        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction, support
