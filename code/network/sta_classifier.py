import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    cuda_index = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.append('../util')
sys.path.append('..')
from loss_util import *
from point_util import *
from base_model_util import *
import pointnet2_model_api as PN2
from pointnet2_ops.pointnet2_utils import QueryAndGroup
# from avg_shape_2.avg_shape_1 import Model as Model_WSLoss


class STA_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.GPN = 512
        self.E_R = PcnEncoder2(out_c=512)
        self.E_A = PcnEncoder2(out_c=512)
        self.D_R = MlpConv(512, [512, 512, 1024, 1024, self.GPN*3])
        self.D_A = MlpConv(512, [512, 512, 1024, 1024, self.GPN*3])

        self.mlp_mirror_ab = MlpConv(512, [128, 128, 2])

        self.mlp_refine_1 = MlpConv(3, [256, 256, 256])
        self.mlp_refine_2 = MlpConv(512, [256, 256, 256])
        self.qg = QueryAndGroup(0.25, 32)
        self.mlp_refine_3 = MlpConv(512, [256, 256, 256])

        self.UPN = 4
        self.mlp_refine_4 = MlpConv(256+256+3, [512, 512, 3*self.UPN])

    
    def get_mirror(self, point, ab):
        __e = 1e-8
        A, B = torch.split(ab, [1, 1], 1)

        x = point[:,:,0:1]
        z = point[:,:,2:3]
@@ -112,110 +112,110 @@ class USSPA_G(nn.Module):

        x = self.upsampling_refine(x)
        point_R_3, point_A_3 = torch.split(x, [B, B], 0)

        other = []
        other.append(point_R_0)
        other.append(input_R_M)

        return f_R, f_A, point_R, point_R_3, point_A, point_A_3, input_R_point_R_0, other


class PointDIS(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        self.encoder = PcnEncoder2(out_c=256)
        self.mlp = MlpConv(256, [64, 64, class_num+1])
    
    def forward(self, point):
        d_p = self.encoder(point)
        d_p = self.mlp(d_p)
        d_p = d_p[:,:,0]
        return d_p


class STA_D(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        self.d_f = MlpConv(512, [64, 64, class_num+1])
        self.d_p = PointDIS(class_num)
    
    def discriminate_feature(self, f):
        d_f = self.d_f(f)
        d_f = d_f[:,:,0]
        return d_f
    
    def forward(self, f_R, f_A, input_R_point_R_0, point_R_3, input_A):
        B = f_R.shape[0]
        f = torch.cat([f_R, f_A], 0)
        d_f = self.discriminate_feature(f)
        d_f_R, d_f_A = torch.split(d_f, [B, B], 0)
        point = torch.cat([input_R_point_R_0, point_R_3, input_A], 0)
        d_p = self.d_p(point)
        d_p_R, d_p_R_3, d_p_A = torch.split(d_p, [B, B, B], 0)

        return d_f_R, d_f_A, d_p_R, d_p_R_3, d_p_A


class STA(nn.Module):
    def __init__(self, class_dict):
        super().__init__()
        self.class_dict = class_dict
        self.class_num = len(list(self.class_dict.keys()))

        self.G = STA_G()
        self.D = STA_D(self.class_num)

        self.loss = staLoss()
        self.loss_test = staLoss_test()
    
    def forward(self, data):
        rc_data, sn_data = data
        input_R = rc_data[0]
        input_A = sn_data[0]

        input_R_label = rc_data[-1]
        input_A_label = sn_data[-1]
        input_R_label = [self.class_dict[x]+1 for x in input_R_label]
        input_A_label = [self.class_dict[x]+1 for x in input_A_label]
        input_R_label = torch.from_numpy(np.array(input_R_label)).cuda().long()
        input_A_label = torch.from_numpy(np.array(input_A_label)).cuda().long()

        f_R, f_A, point_R, point_R_3, point_A, point_A_3, input_R_point_R_0, other = self.G(input_R, input_A)
        d_f_R, d_f_A, d_p_R, d_p_R_3, d_p_A = self.D(f_R, f_A, input_R_point_R_0, point_R_3, input_A)

        other.append(input_R_label)
        other.append(input_A_label)

        return point_R, point_R_3, point_A, point_A_3, input_R_point_R_0, \
            d_f_R, d_f_A, d_p_R, d_p_R_3, d_p_A, \
            input_R, input_A, other


class staLoss(BasicLoss):
    def __init__(self):
        super().__init__()
        self.loss_name = ['loss_g', 'loss_d', 'g_fake_loss', 'g_rsl_2', 'g_rsl_2', 'g_fsl_3', 'density_loss', 'd_fake_loss', 'd_real_loss']
        self.loss_num = len(self.loss_name)
        self.distance = ChamferDistance()
        self.class_loss = nn.CrossEntropyLoss(reduction='none')
    
    def cd(self, p1, p2):
        p2g, g2p = self.distance(p1, p2)
        p2g = torch.mean(p2g, 1)
        g2p = torch.mean(g2p, 1)
        cd = p2g + g2p
        return cd, p2g, g2p
    
    def density_loss(self, x):
        x1 = x.unsqueeze(1)
        x2 = x.unsqueeze(2)
        diff = (x1-x2).norm(dim=-1)
        diff, idx = diff.topk(16, largest=False)
        loss = diff[:,:,1:].mean(2).std(1)
        return loss
    
    def calc_g_fake_loss(self, d):
        __E = 1e-8
        d = torch.softmax(d, -1)
@@ -233,51 +233,51 @@ class USSPALoss(BasicLoss):

        point_R_0 = other[0]
        input_A_label = other[-1]
        input_R_label = torch.zeros([B]).cuda().long()

        g_fake_loss = self.calc_g_fake_loss(d_f_R) + self.calc_g_fake_loss(d_p_R) + self.calc_g_fake_loss(d_p_R_3)

        g_rsl, _, _ = self.cd(point_A, input_A)
        g_rsl_2, _, _ = self.cd(point_A_3, input_A)
        g_fsl, _, _ = self.cd(point_R, input_R_point_R_0)
        g_fsl_2, _, _ = self.cd(point_R_3, input_R_point_R_0)
        _, _, g_fsl_3 = self.cd(point_R_0, input_R)

        density_loss = self.density_loss(point_A) + self.density_loss(point_R)

        loss_g = g_fake_loss + 1e2 * g_rsl + 1e2 * g_rsl_2 + 1e2 * g_fsl + 1e2 * g_fsl_2 + 1e2 * g_fsl_3 + 1e1 * density_loss

        d_fake_loss = self.class_loss(d_f_R, input_R_label) + self.class_loss(d_p_R, input_R_label) + self.class_loss(d_p_R_3, input_R_label)
        d_real_loss = self.class_loss(d_f_A, input_A_label) + self.class_loss(d_p_A, input_A_label)

        loss_d = (d_real_loss+d_fake_loss)/2

        return [loss_g, loss_d, g_fake_loss, g_rsl_2, g_fsl_2, g_fsl_3, density_loss, d_fake_loss, d_real_loss]


class staLoss_test(BasicLoss):
    def __init__(self):
        super().__init__()
        self.loss_name = ['cd', 'fcd_0p001', 'fcd_0p01', 'den_loss', 'acc']
        self.loss_num = len(self.loss_name)
        self.distance = ChamferDistance()
    
    def cd1(self, p1, p2):
        p2g, g2p = self.distance(p1, p2)
        p2g = torch.sqrt(p2g)
        g2p = torch.sqrt(g2p)
        p2g = torch.mean(p2g, 1)
        g2p = torch.mean(g2p, 1)
        cd = (p2g + g2p)/2
        return cd

    def cd2(self, p1, p2):
        p2g, g2p = self.distance(p1, p2)
        p2g = torch.mean(p2g, 1)
        g2p = torch.mean(g2p, 1)
        cd = p2g + g2p
        return cd

    def density_loss(self, x):
        x1 = x.unsqueeze(1)
        x2 = x.unsqueeze(2)
@@ -289,26 +289,26 @@ class USSPALoss_test(BasicLoss):
        return loss, mean
    

    def batch_forward(self, outputs, data):
        __E = 1e-8
        point_R, point_R_3, point_A, point_A_3, input_R_point_R_0, \
        d_f_R, d_f_A, d_p_R, d_p_R_3, d_p_A, \
        input_R, input_A, other = outputs

        gt = data[0][1]
        input_R_label = other[-2]

        cd = self.cd1(point_R_3, gt)
        fcd_0p001 = calc_fcd(point_R_3, gt, a=0.001)
        fcd_0p01 = calc_fcd(point_R_3, gt, a=0.01)

        den_loss, mean = self.density_loss(point_R_3)

        pre_label_3 = torch.argmax(d_p_R_3[:,1:])+1

        acc = (input_R_label == pre_label_3).float()

        return [cd, fcd_0p001, fcd_0p01, den_loss, acc]    

if __name__ == '__main__':
    model = STA()