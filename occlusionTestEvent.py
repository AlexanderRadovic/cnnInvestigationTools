import sys
import caffe
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import lmdb
import argparse
import leveldb
import ROOT
from collections import defaultdict

def plot_heatmap(data, outfile):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.set_xlabel('Plane')
    ax.set_ylabel('Cell')

    imgplot = plt.imshow(np.rot90(data), vmin=-0.1, vmax=0.1, cmap='RdGy', interpolation='none', extent=[0,100,0,80])

    ax.imshow(np.rot90(data), vmin=-0.1, vmax=0.1, cmap='RdGy', interpolation='none', extent=[0,100,0,80])
    ax.imshow(np.rot90(data), vmin=-0.1, vmax=0.1, cmap='RdGy', interpolation='none', extent=[0,100,0,80])
    bar=plt.colorbar()
    bar.set_ticks([])
    plt.savefig(outfile,dpi = 1000)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--leveldb', type=str, required=True)
    parser.add_argument('--event', type=int, required=True)
    args = parser.parse_args()

    count = 0
    correct = 0
    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_gpu()
    db = leveldb.LevelDB(args.leveldb)

    numpy_data=[]
    numpy_images=[]

    cutcount=0
    
    for key, value in db.RangeIter():
        if count > 1000:
            break
    
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        count = count + 1
        out = net.forward_all(data=np.asarray([image]))
        plabel = int(out['prob'][0].argmax(axis=0))

        plabel_major = np.zeros(5)
        plabel_major[0]=out['prob'][0][0]+out['prob'][0][1]+out['prob'][0][2]+out['prob'][0][3]
        plabel_major[1]=out['prob'][0][4]+out['prob'][0][5]+out['prob'][0][6]+out['prob'][0][7]
        plabel_major[2]=out['prob'][0][8]+out['prob'][0][9]+out['prob'][0][10]+out['prob'][0][11]
        plabel_major[3]=out['prob'][0][12]+out['prob'][0][13]
        plabel_major[4]=out['prob'][0][14]

        event_image= np.asarray([image])
        
        if (np.count_nonzero(np.asarray([image])) > 40) & (count == args.event):
            cutcount=cutcount+1
        


            ################################################
            bs, col, s0, s1 = event_image.shape
            square_length=2
            num_classes=15
            num_classes_major=5
            heat_array = np.zeros((s0, s1))
            pad = square_length // 2 + 1
            x_occluded = np.zeros((s0, s1, col, s0, s1))
            x_occluded=x_occluded.astype(np.uint8)

            x_occluded_x= np.zeros((s0, s1, col, s0, s1))
            x_occluded_x=x_occluded.astype(np.uint8)

            x_occluded_y= np.zeros((s0, s1, col, s0, s1))
            x_occluded_y=x_occluded.astype(np.uint8)
            
            probs = np.zeros((num_classes,s0, s1))
            probs_major = np.zeros((num_classes_major, s0, s1))

            probs_x = np.zeros((num_classes,s0, s1))
            probs_major_x = np.zeros((num_classes_major, s0, s1))

            probs_y = np.zeros((num_classes,s0, s1))
            probs_major_y = np.zeros((num_classes_major, s0, s1))

            
            #print event_image.shape
            # generate occluded images
            for i in range(s0):
                # batch s1 occluded images for faster prediction
                for j in range(s1):
                    x_occluded[i,j] = event_image
                    x_occluded_x[i,j] = event_image
                    x_occluded_y[i,j] = event_image
                    for k in range(-square_length,square_length+1):
                        for l in range(-square_length,square_length+1):
                            if ((i+l) < 0) or ((i+l) > (s0-1)):
                                continue
                            if ((j+k) < 0) or ((j+k) > (s1-1)):
                                continue
                            # print "k,l:"
                            # print k
                            # print l
                            # print "i,j:"
                            # print i
                            # print j
                            # print "i,j shift:"
                            # print i+l
                            # print j+k
                            # print x_occluded[i,j,0,i+l,j+k]
                            # print x_occluded[i,j,1,i+l,j+k]
                            # print x_occluded_x[i,j,0,i+l,j+k]
                            # print x_occluded_y[i,j,1,i+l,j+k]
                            x_occluded[i,j,0,i+l,j+k] = 0
                            x_occluded[i,j,1,i+l,j+k] = 0
                            x_occluded_x[i,j,0,i+l,j+k] = 0
                            x_occluded_y[i,j,1,i+l,j+k] = 0
                            # print x_occluded[i,j,0,i+l,j+k]
                            # print x_occluded[i,j,1,i+l,j+k]
                            # print x_occluded_x[i,j,0,i+l,j+k]
                            # print x_occluded_y[i,j,1,i+l,j+k]
                            
                            
                    reformat = np.zeros((col, s0, s1))
                    reformat = reformat.astype(np.uint8)
                    out_occluded = net.forward_all(data=np.asarray([x_occluded[i,j]]))
                    out_occluded_x = net.forward_all(data=np.asarray([x_occluded_x[i,j]]))
                    out_occluded_y = net.forward_all(data=np.asarray([x_occluded_y[i,j]]))
                    plabel_occluded = int(out_occluded['prob'][0].argmax(axis=0))

                    for q in range(num_classes):
                        probs[q,i,j]=out_occluded['prob'][0][q]

                    probs_major[0,i,j]=out_occluded['prob'][0][0]+out_occluded['prob'][0][1]+out_occluded['prob'][0][2]+out_occluded['prob'][0][3]
                    probs_major[1,i,j]=out_occluded['prob'][0][4]+out_occluded['prob'][0][5]+out_occluded['prob'][0][6]+out_occluded['prob'][0][7]
                    probs_major[2,i,j]=out_occluded['prob'][0][8]+out_occluded['prob'][0][9]+out_occluded['prob'][0][10]+out_occluded['prob'][0][11]
                    probs_major[3,i,j]=out_occluded['prob'][0][12]+out_occluded['prob'][0][13]
                    probs_major[4,i,j]=out_occluded['prob'][0][14]

                    for q in range(num_classes):
                        probs_x[q,i,j]=out_occluded_x['prob'][0][q]

                    probs_major_x[0,i,j]=out_occluded_x['prob'][0][0]+out_occluded_x['prob'][0][1]+out_occluded_x['prob'][0][2]+out_occluded_x['prob'][0][3]
                    probs_major_x[1,i,j]=out_occluded_x['prob'][0][4]+out_occluded_x['prob'][0][5]+out_occluded_x['prob'][0][6]+out_occluded_x['prob'][0][7]
                    probs_major_x[2,i,j]=out_occluded_x['prob'][0][8]+out_occluded_x['prob'][0][9]+out_occluded_x['prob'][0][10]+out_occluded_x['prob'][0][11]
                    probs_major_x[3,i,j]=out_occluded_x['prob'][0][12]+out_occluded_x['prob'][0][13]
                    probs_major_x[4,i,j]=out_occluded_x['prob'][0][14]

                    for q in range(num_classes):
                        probs_y[q,i,j]=out_occluded_y['prob'][0][q]

                    probs_major_y[0,i,j]=out_occluded_y['prob'][0][0]+out_occluded_y['prob'][0][1]+out_occluded_y['prob'][0][2]+out_occluded_y['prob'][0][3]
                    probs_major_y[1,i,j]=out_occluded_y['prob'][0][4]+out_occluded_y['prob'][0][5]+out_occluded_y['prob'][0][6]+out_occluded_y['prob'][0][7]
                    probs_major_y[2,i,j]=out_occluded_y['prob'][0][8]+out_occluded_y['prob'][0][9]+out_occluded_y['prob'][0][10]+out_occluded_y['prob'][0][11]
                    probs_major_y[3,i,j]=out_occluded_y['prob'][0][12]+out_occluded_y['prob'][0][13]
                    probs_major_y[4,i,j]=out_occluded_y['prob'][0][14]

                    
            for q in range(num_classes): 
                plot_heatmap(probs[q],'occtest_truetype'+str(label)+'_caltype'+str(plabel)+'_event'+str(count)+'_pidoutput'+str(q)+'.pdf')
                
            for q in range(num_classes_major): 
                plot_heatmap(probs_major[q]-plabel_major[q],'occtest_truetype'+str(label)+'_caltype'+str(plabel)+'_event'+str(count)+'_pidoutput_major'+str(q)+'.pdf')

            for q in range(num_classes): 
                plot_heatmap(probs_x[q],'occtest_truetype'+str(label)+'_caltype'+str(plabel)+'_event'+str(count)+'_pidoutput'+str(q)+'_x.pdf')
                
            for q in range(num_classes_major): 
                plot_heatmap(probs_major_x[q]-plabel_major[q],'occtest_truetype'+str(label)+'_caltype'+str(plabel)+'_event'+str(count)+'_pidoutput_major'+str(q)+'_x.pdf')

            for q in range(num_classes): 
                plot_heatmap(probs_y[q],'occtest_truetype'+str(label)+'_caltype'+str(plabel)+'_event'+str(count)+'_pidoutput'+str(q)+'_y.pdf')
                
            for q in range(num_classes_major): 
                plot_heatmap(probs_major_y[q]-plabel_major[q],'occtest_truetype'+str(label)+'_caltype'+str(plabel)+'_event'+str(count)+'_pidoutput_major'+str(q)+'_y.pdf')

            
            imageswap=np.swapaxes(event_image.reshape(np.delete(event_image.shape,0)),0,2)
            x,y=np.dsplit(imageswap,2)
            x=x.swapaxes(0,1)
            y=y.swapaxes(0,1)

            x=np.rot90(x.squeeze())
            y=np.rot90(y.squeeze())
            
            fig, ax = plt.subplots(figsize=(6,5))
            ax.set_xlabel('Plane')
            ax.set_ylabel('Cell')
        
            ax.imshow(x, cmap='gist_heat_r', interpolation='none', extent=[0,100,0,80])
            ax.imshow(x, cmap='gist_heat_r', interpolation='none', extent=[0,100,0,80])

            #plt.colorbar(fig)
            plt.savefig('occtest_view_truetype'+str(label)+'_caltype'+str(plabel)+'_event'+str(count)+'_x.pdf',dpi = 1000)
            
            fig2, ax2 = plt.subplots(figsize=(6,5))
            ax2.set_xlabel('Plane')
            ax2.set_ylabel('Cell')
            ax2.imshow(y, cmap='gist_heat_r', interpolation='none', extent=[0,100,0,80])
            ax2.imshow(y, cmap='gist_heat_r', interpolation='none', extent=[0,100,0,80])
            #plt.colorbar(fig2)
            print 'occtest_view_truetype'+str(label)+'_caltype'+str(plabel)+'_event'+str(count)+'_y.pdf'
            plt.savefig('occtest_view_truetype'+str(label)+'_caltype'+str(plabel)+'_event'+str(count)+'_y.pdf',dpi = 1000)


    
    
