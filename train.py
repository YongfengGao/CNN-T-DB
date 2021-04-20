from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from numpy.random import normal

from math import sqrt
import input_data_triple  as input_data
import numpy as np
import math
import os
import build_model
from matplotlib import pyplot as plt
import glob
import random
import time
# Training settings



parser = argparse.ArgumentParser(description='Vanilla 3D-CNN')

parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 30)')

parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--iteration', type=int, default=15, metavar='N',
                    help='number of iterations to train (default: 60000)')

parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='learning rate (default: 1e-7)')
#lung learning rate:1e-2
#muscle learning rate:1e-2
#fat learning rate:1e-2
#bone learning rate:1e-2

parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--weight_decay', type=float, default=0.005, metavar='M',
                    help='weight_decay (default: 0.005)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--save', type=str,  default='model_lung/',
                    help='path to save the final model')

parser.add_argument('--log', type=str,  default='log/',
                    help='path to the log information')


def training(args):

    for part in ["lung"]:
        Vanilla_3D = build_model.basic().cuda(0)
        Vanilla_3D.train(True)
        

        #optimizer =optim.RMSprop(Vanilla_3Dt.parameters(), lr=args.lr)
        optimizer = optim.Adam(Vanilla_3D.parameters(), lr=args.lr)

        criterion = nn.L1Loss(size_average=True).cuda(0)
        loss_to_draw = []
        lines=glob.glob("patch\\"+"\\*.mat")

        print(part," : ",len(lines))
        for step_index in range(args.iteration):
            for idx in range(len(lines)//args.batch_size-1):
                patch=lines[idx*args.batch_size:(idx+1)*args.batch_size]

                train_image,train_label = input_data.ReadDataPixel(patch) # patch to pixel model
                train_image = np.transpose(train_image,(0,3,1,2))

                if train_image.shape[0] is 0:
                        continue

                train_image = torch.from_numpy(train_image)
                train_label = torch.from_numpy(train_label)
                

                train_image, train_label = train_image.cuda(0), train_label.cuda(0)
                train_image, train_label = Variable(train_image), Variable(train_label)
                #train_label=train_label.long()
                mask = Vanilla_3D(train_image)
                #loss= criterion((mask+1)*100000, train_label)
                loss= criterion(mask, train_label)
                loss_to_draw.append(loss.data.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step_index%10000) == 0:
                    lr = args.lr * (0.1 ** (step_index // 20000))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                if ((idx+1)%30) == 0:
                    plt_save_dir = "./loss_imgs"
                    plt_save_img_name = str(step_index) +"_"+str(idx)+ '.png'
                    plt.clf()
                    plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
                    plt.grid(True)
                    #plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))


                if (idx%10) ==0:
                    print('Step: {}, Step2: {}, loss: {}\n'.format(step_index, idx,loss.data.item()))

    with open("models\\" + part, 'wb') as f:
        torch.save(Vanilla_3D, f)


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    training(args)

if __name__ == '__main__':
    main()
