#Takes numpy array of network output, best guess label, and true label as input. Makes PCA and TSNE representations of the network output using network output. Labels used to color code points in the PCA/TSNE space.

import sys
import caffe
#import matplotlib as plt
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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()
    
    npy_data=np.load('data'+args.input+'.npy')

    ntrain=7000
    features=npy_data[:ntrain,:-2]
    features_svd=TruncatedSVD(n_components=50,random_state=0).fit_transform(features)
    print features.shape
    print features_svd.shape
    plabel=npy_data[:ntrain,-2]
    print plabel.shape
    tlabel=npy_data[:ntrain,-1]
    print tlabel.shape

    features_TSNE=TSNE(learning_rate=100,init='pca',perplexity=50,n_iter_without_progress=100,verbose=2).fit_transform(features)#_svd)    

    features_PCA=PCA().fit_transform(features)#_svd)

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    scat = plt.scatter(features_PCA[:,0],features_PCA[:,1],c=plabel,cmap=plt.get_cmap("jet",15),vmin=0,vmax=15)
    cb = plt.colorbar(scat,ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5])
    plt.title('PCA, Reco Labels')
    cb.set_ticklabels([r'$\nu_{\mu} $ $CC$ $ QE$',
                       r'$\nu_{\mu} $ $CC$ $ RES$',
                       r'$\nu_{\mu} $ $CC$ $ DIS$',
                       r'$\nu_{\mu} $ $CC$ $ COH$',
                       r'$\nu_{e} $ $CC$ $ QE$',
                       r'$\nu_{e} $ $CC$ $ RES$',
                       r'$\nu_{e} $ $CC$ $ DIS$',
                       r'$\nu_{e} $ $CC$ $ COH$',
                       r'$\nu_{\tau} $ $CC$ $ QE$',
                       r'$\nu_{\tau} $ $CC$ $ RES$',
                       r'$\nu_{\tau} $ $CC$ $ DIS$',
                       r'$\nu_{\tau} $ $CC$ $ COH$',
                       r'$\nu $ $on$ $ e$',
                       r'$\nu_{x} $ $NC$',
                       r'$Cosmics$'])
    
    plt.subplot(122)
    scat2 = plt.scatter(features_TSNE[:,0],features_TSNE[:,1],c=plabel,cmap=plt.get_cmap("jet",15),vmin=0,vmax=15)
    cb2 = plt.colorbar(scat,ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5])
    plt.title('t-SNE, Reco Labels')
    cb2.set_ticklabels([r'$\nu_{\mu} $ $CC$ $ QE$',
                        r'$\nu_{\mu} $ $CC$ $ RES$',
                        r'$\nu_{\mu} $ $CC$ $ DIS$',
                        r'$\nu_{\mu} $ $CC$ $ COH$',
                        r'$\nu_{e} $ $CC$ $ QE$',
                        r'$\nu_{e} $ $CC$ $ RES$',
                        r'$\nu_{e} $ $CC$ $ DIS$',
                        r'$\nu_{e} $ $CC$ $ COH$',
                        r'$\nu_{\tau} $ $CC$ $ QE$',
                        r'$\nu_{\tau} $ $CC$ $ RES$',
                        r'$\nu_{\tau} $ $CC$ $ DIS$',
                        r'$\nu_{\tau} $ $CC$ $ COH$',
                        r'$\nu $ $on$ $ e$',
                        r'$\nu_{x} $ $NC$',
                        r'$Cosmics$'])
       
    plt.show()

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    scat3 = plt.scatter(features_TSNE[:,0],features_TSNE[:,1],c=plabel,cmap=plt.get_cmap("jet",15),vmin=0,vmax=15)
    cb3 = plt.colorbar(scat3,ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5])
    plt.title('Reco Labels')
    cb3.set_ticklabels([r'$\nu_{\mu} $ $CC$ $ QE$',
                       r'$\nu_{\mu} $ $CC$ $ RES$',
                       r'$\nu_{\mu} $ $CC$ $ DIS$',
                       r'$\nu_{\mu} $ $CC$ $ COH$',
                       r'$\nu_{e} $ $CC$ $ QE$',
                       r'$\nu_{e} $ $CC$ $ RES$',
                       r'$\nu_{e} $ $CC$ $ DIS$',
                       r'$\nu_{e} $ $CC$ $ COH$',
                       r'$\nu_{\tau} $ $CC$ $ QE$',
                       r'$\nu_{\tau} $ $CC$ $ RES$',
                       r'$\nu_{\tau} $ $CC$ $ DIS$',
                       r'$\nu_{\tau} $ $CC$ $ COH$',
                       r'$\nu $ $on$ $ e$',
                       r'$\nu_{x} $ $NC$',
                       r'$Cosmics$'])

    plt.subplot(122)
    scat4 = plt.scatter(features_TSNE[:,0],features_TSNE[:,1],c=tlabel,cmap=plt.get_cmap("jet",15),vmin=0,vmax=15)
    cb4 = plt.colorbar(scat,ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5])
    plt.title('Truth Labels')
    cb4.set_ticklabels([r'$\nu_{\mu} $ $CC$ $ QE$',
                       r'$\nu_{\mu} $ $CC$ $ RES$',
                       r'$\nu_{\mu} $ $CC$ $ DIS$',
                       r'$\nu_{\mu} $ $CC$ $ COH$',
                       r'$\nu_{e} $ $CC$ $ QE$',
                       r'$\nu_{e} $ $CC$ $ RES$',
                       r'$\nu_{e} $ $CC$ $ DIS$',
                       r'$\nu_{e} $ $CC$ $ COH$',
                       r'$\nu_{\tau} $ $CC$ $ QE$',
                       r'$\nu_{\tau} $ $CC$ $ RES$',
                       r'$\nu_{\tau} $ $CC$ $ DIS$',
                       r'$\nu_{\tau} $ $CC$ $ COH$',
                       r'$\nu $ $on$ $ e$',
                       r'$\nu_{x} $ $NC$',
                       r'$Cosmics$'])

    plt.show()

    
