"""
    Copyright (c) 2018-present. Ben Athiwaratkun
    All rights reserved.
    
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree.
"""
# A sample script for qualitative evaluation by querying nearest neighbors
import sys, os
sys.path.append("../")
sys.path.append(".")
import embeval
import numpy as np
import pickle
import argparse
## First, let's look at the nearest neighbors

def show_nn(modelname, multi):
  print "Basename = ", modelname
  ft = embeval.get_fts([modelname], multi)[0]
  for word in ['rock', 'star', 'cell', 'left']:
    print "Nearest Neighbors for {}, cluster 0".format(word)
    ft.show_nearest_neighbors_single(word, cl=0)
    if multi:
      print "Nearest Neighbors for {}, cluster 1".format(word)
      ft.show_nearest_neighbors(word, cl=1)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--modelname', default='', type=str, help="The model to be evaluated. For instance, the files 'modelname.in', 'modelname.bin', etc should exist.")
  parser.add_argument('--multi', default=1, type=int, help="Whether this is a multisense model")
  args = parser.parse_args()

  show_nn(modelname=args.modelname, multi=args.multi)