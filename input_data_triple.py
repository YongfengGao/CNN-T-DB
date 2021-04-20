from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io
import scipy.misc
import gzip
import os
import tempfile
import array
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import PIL.Image as Image
import random
import numpy as np
import time
import glob
from matplotlib import pyplot as plt
#from scipy.misc import imresize
import matplotlib.pyplot as plt
import glob,os


IMAGE_SIZE = 7



'''
    Read Training dataset!
'''
def ReadDataPixel(lines):
  data = []
  data2=[]

  # batch_index = 0
  for line in lines:
    line = line.strip()
    if "full" in line: # full means the boundary pathes are included in this folder
      continue
    image_name = str(line)
    tmp_data=scipy.io.loadmat(image_name)['a']
    y=tmp_data[3:4,3:4].copy()
    #print(y,np.shape(y))
    tmp_data[3:4,3:4]=0.0
    tmp_data=tmp_data.reshape(tmp_data.shape[0],tmp_data.shape[1],1)
    y=y.reshape(y.shape[0],y.shape[1],1)
    # print(y,np.shape(y))
    # print(y.shape)
    #y=y.reshape(y.shape[0],1)
    #print(y.shape,y)

    data.append(tmp_data)
    data2.append(y)

  np_arr_data = np.array(data)
  np_arr_data = np_arr_data.astype(numpy.float32)

  np_arr_data2 = np.array(data2)
  np_arr_data2 = np_arr_data2.astype(numpy.float32)
  return np_arr_data,np_arr_data2


