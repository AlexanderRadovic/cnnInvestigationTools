#Very similiar to tsneSciKit but also draws example events on top of the TSNE space. Relies on the file of "images" being ordered the same way as the file of network outputs/labels.

import sys
import caffe
import matplotlib.pyplot as plt
import numpy as np
import lmdb
import argparse
import leveldb
import ROOT
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict

if __name__ == "__main__":
    #command line argument for input numpy file, and whether or not to draw the overlaid example images
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--embedd', type=str, required=True)

    args = parser.parse_args()

    npy_data=np.load('data'+args.input+'.npy')
    npy_images=np.load('images'+args.input+'.npy')

    #decide on size of sample to use in the TSNE, create feature array and label arrays
    #assumes the data is formatted by network outputs, followed by the final prediction, and ending with the true label
    ntrain=14000
    features=npy_data[:ntrain,:-2]
    features_svd=TruncatedSVD(n_components=50,random_state=0).fit_transform(features)
    print features.shape
    print features_svd.shape
    plabel=npy_data[:ntrain,-2]
    print plabel.shape
    tlabel=npy_data[:ntrain,-1]
    print tlabel.shape


    #build a tsne map, starting from a pca intialisation proved essential for our sample
    features_TSNE=TSNE(learning_rate=100,init='pca',perplexity=50,n_iter_without_progress=100,verbose=2).fit_transform(features)#_svd)

    fig = plt.figure(figsize=(50, 50))
    ax = plt.axes(frameon=False)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)    
    scat=plt.scatter(features_TSNE[:,0],features_TSNE[:,1],c=tlabel,cmap=plt.get_cmap("jet",14),vmin=0,vmax=14)

    #code to plot a random sampling of events over our tsne distribution
    if args.embedd == 'true':

        from matplotlib import offsetbox

        shown_images = np.array([[15., 15.]])
        indices = np.arange(features_TSNE.shape[0])
        min_dist = 4.5

        for i in indices[:ntrain]:
            #try to plot events which are distant in the feature space compared to those than have come before,
            #skip events which are too close, play with min_dist to make the overlay more or less crowded
            dist = np.sum((features_TSNE[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist:
                continue

            data=npy_images[i,:]
            dataswap=np.swapaxes(data.reshape(np.delete(data.shape,0)),0,2)
            x,y=np.dsplit(dataswap,2)
            x=x.swapaxes(0,1)
            y=y.swapaxes(0,1)

            x=np.rot90(x.squeeze())
            y=np.rot90(y.squeeze())

            #append the image we are about to plot so we can make sure the next one is disimiliar 
            shown_images = np.r_[shown_images, [features_TSNE[i]]]

            #draw the more active of the x and y views
            if np.count_nonzero(x) > np.count_nonzero(y):
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(x,cmap='gist_heat_r', interpolation='none'), features_TSNE[i])
            else:
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(y,cmap='gist_heat_r', interpolation='none'), features_TSNE[i])

            
            ax.add_artist(imagebox)

    cb = plt.colorbar(scat,ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5], shrink=0.95)
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
                       r'$\nu_{x} $ $NC$'])

    plt.show()
    plt.savefig('tsneShowCase.pdf',dpi = 1000)
