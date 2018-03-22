import matplotlib.pyplot as plt
import numpy as np

#simple plotting macro that uses a heatmap color scheme and avoids interpolation
def plot_heatmap(data, outfile):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_xlabel('Plane')
    ax.set_ylabel('Cell')

    imgplot = plt.imshow(np.rot90(data), cmap='gist_heat_r', interpolation='none', extent=[0,100,0,80])

    ax.imshow(np.rot90(data), cmap='gist_heat_r', interpolation='none', extent=[0,100,0,80])
    ax.imshow(np.rot90(data), cmap='gist_heat_r', interpolation='none', extent=[0,100,0,80])
    plt.colorbar()

    plt.savefig(outfile,dpi = 1000)

#edits to color style to make the story clearer on a large projector
def plot_heatmap_presentation(data, outfile):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_xlabel('Plane')
    ax.set_ylabel('Cell')

    imgplot = plt.imshow(np.rot90(data), vmin=-0.1, vmax=0.1, cmap='RdGy', interpolation='none', extent=[0,100,0,80])

    ax.imshow(np.rot90(data), vmin=-0.1, vmax=0.1, cmap='RdGy', interpolation='none', extent=[0,100,0,80])
    ax.imshow(np.rot90(data), vmin=-0.1, vmax=0.1, cmap='RdGy', interpolation='none', extent=[0,100,0,80])
    bar=plt.colorbar()
    bar.set_ticks([])
    plt.savefig(outfile,dpi = 1000)

#a scatter plot to compare different tsne projections, coloring each point based on either a true or predicted label
def plot_tsne_comparison(exampleOne, exampleTwo, dataLabelOne, dataLabelTwo, titleOne, titleTwo, outFile)

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    scat = plt.scatter(exampleOne[:,0],exampleOne[:,1],c=dataLabelOne,cmap=plt.get_cmap("jet",15),vmin=0,vmax=15)
    cb = plt.colorbar(scat,ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5])
    plt.title(titleOne)
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
    scat2 = plt.scatter(exampleTwo[:,0],exampleTwo[:,1],c=dataLabelTwo,cmap=plt.get_cmap("jet",15),vmin=0,vmax=15)
    cb2 = plt.colorbar(scat,ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5])
    plt.title(titleTwo)
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
    plt.savefig(outFile,dpi = 1000)



