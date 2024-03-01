import torch
import torch.nn as nn
import torch.linalg
import cv2
import numpy as np

def rt2edu_matrix(rot, u, v):
    Euclidean_matrix = torch.tensor([[torch.cos(rot), -torch.sin(rot), u],
                                    [torch.sin(rot), torch.cos(rot), v],
                                    [torch.zeros_like(u), torch.zeros_like(u), torch.ones_like(u)]], device=rot.device)
    return Euclidean_matrix


#Least-Squares Fitting of Two 3-D Point Sets
class Least_Squares(nn.Module):
    def __init__(self, all_epoch):
        super(Least_Squares, self).__init__()
        self.all_epoch = all_epoch
    
    def forward(self, pstA, pstB):
        #pstA [B,num,2]
        #pstB [B,num,2]
        #x [B,num]
        B, num, _ = pstA.size()
        G_A = torch.mean(pstA, axis=1)
        G_B = torch.mean(pstB, axis=1)

        Am = pstA - G_A[:,None,:]
        Bm = pstB - G_B[:,None,:]

        H = Am.permute(0,2,1) @ Bm
        U, S, V = torch.svd(H)
        # print(U@torch.diag(S[0])@V.permute(0,2,1))
        R = V @ U.permute(0,2,1)
        theta = torch.zeros((B,1),device = pstA.device)
        for i in range(B):
            if torch.det(R[i]) < 0:
                # print("det(R) < R, reflection detected!, correcting for it ...")
                V[i,:,1] *= -1
                R[i] = V[i] @ U[i].T 
            theta[i,0] = torch.arccos((torch.trace(R[i]))/2)*R[i,1,0]/torch.abs(R[i,0,1])

        G_A = G_A.unsqueeze(-1)
        G_B = G_B.unsqueeze(-1)
        t = -R @ G_A + G_B
        return theta, t[:,0], t[:,1]


def rigid_transform_3D(A, B):
    assert A.shape == B.shape


    # if num_rows != 2:
    #     raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    # num_rows, num_cols = B.shape
    # if num_rows != 2:
    #     raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    # print(U@np.eye(2)*S@Vt)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[1,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B
    theta = np.arccos((np.trace(R))/2)*R[1,0]/np.abs(R[0,1])
    return theta, R, t


#Least-Squares Fitting of Two 3-D Point Sets
class Least_Squares_weight(nn.Module):
    def __init__(self, all_epoch):
        super(Least_Squares_weight, self).__init__()
        self.all_epoch = all_epoch
    
    def forward(self, pstA, pstB, weight):
        #pstA [B,num,2]
        #pstB [B,num,2]
        #x [B,num]
        B, num, _ = pstA.size()

        ws = weight.sum()
        G_A = (pstA * weight).sum(axis=1)/ws
        G_B = (pstB * weight).sum(axis=1)/ws

        Am = (pstA - G_A[:,None,:])*weight
        Bm = (pstB - G_B[:,None,:])*weight

        H = Am.permute(0,2,1) @ Bm
        U, S, V = torch.svd(H)
        # print(U@torch.diag(S[0])@V.permute(0,2,1))
        R = V @ U.permute(0,2,1)
        theta = torch.zeros((B,1),device = pstA.device)
        for i in range(B):
            if torch.det(R[i]) < 0:
                # print("det(R) < R, reflection detected!, correcting for it ...")
                V[i,:,1] *= -1
                R[i] = V[i] @ U[i].T 
            theta[i,0] = torch.arccos((torch.trace(R[i]))/2)*R[i,1,0]/torch.abs(R[i,0,1])

        G_A = G_A.unsqueeze(-1)
        G_B = G_B.unsqueeze(-1)
        t = -R @ G_A + G_B
        return theta, t[:,0], t[:,1]
    
# pstA = torch.tensor([[[],[]], [[],[]]])
if __name__ == '__main__':
    num = 3 
    B = 5

    pstA = torch.rand((B,3,2))*1000
    pstB = torch.rand((B,3,2))*1000
    Least_Squares = Least_Squares(50)
    theta, R, t = Least_Squares(pstA, pstB)
    print("my")
    print(theta)
    # print(R)
    print(t)
    for i in range(B):
        psta_numpy = np.array(pstA[i].T)
        pstb_numpy = np.array(pstB[i].T)
        theta, R, t = rigid_transform_3D(psta_numpy, pstb_numpy)
        R_new = np.eye(3)
        for i in range(2):
            for j in range(2):
                R_new[i,j] = R[i, j]
        print("T")
        print("cv2",cv2.Rodrigues(R_new)[0][2])
        print(theta)
        # print(R)
        print(t)



