#Takes numpy array of network output, best guess label, and true label as input. 
#Makes PCA and TSNE representations of the network output using network output. 
#Labels used to color code points in the PCA/TSNE space.

import sys
import caffe
import matplotlib.pyplot as plt
import numpy as np
import lmdb
import argparse
import leveldb
import ROOT
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import plotFunctions

if __name__ == "__main__":
    #command line argument for input numpy file
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()
    
    npy_data=np.load(args.input+'.npy')

    #decide on size of sample to use in the TSNE, create feature array and label arrays
    #assumes the data is formatted by network outputs, followed by the final prediction, and ending with the true label
    ntrain=7000
    features=npy_data[:ntrain,:-2]
    features_svd=TruncatedSVD(n_components=50,random_state=0).fit_transform(features)
    plabel=npy_data[:ntrain,-2]
    tlabel=npy_data[:ntrain,-1]
    
    #build a tsne map, starting from a pca intialisation proved essential for our sample
    features_TSNE=TSNE(learning_rate=100,init='pca',perplexity=50,n_iter_without_progress=100,verbose=2).fit_transform(features)#_svd)    
    #compare to pca to get a sense of how much tsne is adding
    features_PCA=PCA().fit_transform(features)

    plotFunctions.plot_tsne_comparison(features_PCA, features_TSNE, plabel, plabel, 'PCA, Reco Labels', 'TSNE, Reco Labels', 'pcaVstsne_recoL.png')
    plotFunctions.plot_tsne_comparison(features_PCA, features_TSNE, plabel, tlabel, 'TSNE, Reco Labels', 'TSNE, True Labels', 'recoTsneVsTruetsne.png')


    
