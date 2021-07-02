import os,glob
import sys
import argparse
import numpy as np
from scipy.io import savemat,loadmat
import torch
from torch.autograd import Variable
import struct
from shutil import rmtree
from matplotlib import pyplot as plt
from numpy import *


def testing():

             recon = np.zeros((512,512))
             for part,weight in zip(['lung',],[0.045]):
                with open("./models/"+part, 'rb') as f:  #please specifiy the model name here 
                    Vanilla_3D = torch.load(f).cuda(0)
                Vanilla_3D.eval()
               
                # specifiy the location of input patches
                folder=glob.glob("D:\\*"+"\\*.mat")


                for f in folder:
                    #print(f)
                    if 'full' in f:
                        continue
                    coord = f.split("\\")[-1].split(".")[0].split("-")
                    tmp_data = loadmat(f)['a']
                    # y = tmp_data[3:4,3:4].copy()
                    
                    tmp_data[3:4,3:4] = 0.0
                    tmp_data = tmp_data.reshape(
                        tmp_data.shape[0], tmp_data.shape[1], 1)

                    np_arr_data = np.array([tmp_data])

                    test_image = np_arr_data.astype(np.float32)
                    #print(test_image.shape)
                    test_image = np.transpose(test_image, (0, 3, 1, 2))
                    test_image = torch.from_numpy(test_image)

                    test_image = test_image.cuda(0)
                    test_image = Variable(test_image)

                    pred = Vanilla_3D(test_image) #mid,

                    ps = pred.cpu().data.numpy()
                    ps = np.squeeze(ps, (0, 1))

                    a = int(coord[0])
                    b = int(coord[1])

                    #residual[a, b] = float(ps[3][3])
                    recon[a, b] = float(ps)*weight
                recon = recon  
             savemat('result.mat',mdict={'a':recon})
  
def main(): 
    print("start testing......")
    testing()

if __name__ == '__main__':
    main()
    sys.stdout.write("Done")



