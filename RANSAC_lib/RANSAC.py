import torch
import torch.nn as nn
import random
from RANSAC_lib.euclidean_trans import Least_Squares

class RANSAC(nn.Module):
    def __init__(self, thresh):
        super(RANSAC, self).__init__()
        self.thresh = thresh
        self.epoch = 1000
        self.euclidean_trans = Least_Squares(50)
        self.nogood = 0
        

    def geometricDistance(self, pstA, pstB, E):
        #! batch_si
        pstA = torch.cat((pstA, torch.ones((pstA.size()[0], pstA.size()[1], 1), device=pstA.device)),-1)
        pstA = pstA.unsqueeze(-1)
        pstA_ = E@pstA[0]
        # pstA_ = pstA_/pstA_[:,2]

        pstB = torch.cat((pstB, torch.ones((pstB.size()[0], pstB.size()[1], 1), device=pstB.device)),-1)
        pstB = pstB.unsqueeze(-1)
        error = pstB[0]-pstA_
        return torch.norm(error,dim=1)

    def forward(self, pstA, pstB):
        #batch_size = 1
        #pstA: [B, N, 2]
        #ptsB: [B, N, 2]
        finalE = None
        maxInliers = torch.zeros(0)
        B, N, _ = pstA.size()
        for i in range(self.epoch):
            corr_num1 = random.randrange(0, N)
            corr_num2 = random.randrange(0, N)
            corr_num3 = random.randrange(0, N)
            corr_num4 = random.randrange(0, N)
            corr_num5 = random.randrange(0, N)
            corr1 = pstA[:,[corr_num1,corr_num2,corr_num3,corr_num4,corr_num5], :]
            corr2 = pstB[:,[corr_num1,corr_num2,corr_num3,corr_num4,corr_num5], :]
            Euclidean = self.euclidean_trans(corr1, corr2)
            rot = Euclidean[0]
            u = Euclidean[1]
            v = Euclidean[2]
            #batch_size = 1
            Euclidean_matrix = torch.tensor([[torch.cos(rot), -torch.sin(rot), u],
                                             [torch.sin(rot), torch.cos(rot), v],
                                             [torch.zeros_like(u), torch.zeros_like(u), torch.ones_like(u)]], device=pstA.device)

            inliers = 0

            dis = self.geometricDistance(pstA, pstB, Euclidean_matrix)
            inliers = torch.le(dis,10)

            if inliers.sum(0) > maxInliers.sum(0):
                maxInliers = inliers

            if maxInliers.sum(0) > (N * self.thresh):
                corr1 = pstA[:,maxInliers[:,0], :]
                corr2 = pstB[:,maxInliers[:,0], :]
                Euclidean = self.euclidean_trans(corr1, corr2)
                finalE = Euclidean
                return finalE,Euclidean_matrix
            
        self.nogood += 1
        print("no good!!", self.nogood)
        corr1 = pstA[:,maxInliers[:,0], :]
        corr2 = pstB[:,maxInliers[:,0], :]
        Euclidean = self.euclidean_trans(corr1, corr2)
        finalE = Euclidean
        rot = Euclidean[0]
        u = Euclidean[1]
        v = Euclidean[2]
        Euclidean_matrix = torch.tensor([[torch.cos(rot), -torch.sin(rot), u],
                                            [torch.sin(rot), torch.cos(rot), v],
                                            [torch.zeros_like(u), torch.zeros_like(u), torch.ones_like(u)]], device=pstA.device)
        return finalE,Euclidean_matrix


