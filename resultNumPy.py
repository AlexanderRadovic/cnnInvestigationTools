#Create numpy file of network outputs and true labels, and a file of simple arrays of the input "images"
#Used as an input to the tsne macros.

import sys
import caffe
import matplotlib
import numpy as np
import lmdb
import argparse
import leveldb
import ROOT
from collections import defaultdict

if __name__ == "__main__":
    #take network and data information from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--leveldb', type=str, required=True)
    args = parser.parse_args()

    count = 0
    correct = 0
    matrix = defaultdict(int) # (real,pred) -> int
    labels_set = set()

    #setup caffe network, set leveldb address
    net = caffe.Net(args.proto, args.model, caffe.TEST)
    caffe.set_mode_gpu()
    db = leveldb.LevelDB(args.leveldb)

    numpy_data=[]
    numpy_images=[]

    cutcount=0
    
    #loop over events, saving network outputs and inputs in a numpy array
    for key, value in db.RangeIter():
        if count > 100000:
            break
    
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        count = count + 1
        
        #for events with enough activity to be interesting, store data for tsne plots
        if np.count_nonzero(np.asarray([image])) > 40:
            cutcount=cutcount+1
        
            out = net.forward_all(data=np.asarray([image]))
            plabel = int(out['prob'][0].argmax(axis=0))
            featuremap=net.blobs['pool5/7x7_s1'].data
            featuremap=featuremap.flatten()
            featuremap=featuremap.astype(dtype=float)
            

        
            iscorrect = label == plabel
            correct = correct + (1 if iscorrect else 0)
            matrix[(label, plabel)] += 1
            labels_set.update([label, plabel])

            featuremapwithlabel=np.append(featuremap,plabel)
            featuremapwithbothlabels=np.append(featuremapwithlabel,label)
        
            numpy_data.append(featuremapwithbothlabels)
            numpy_images.append(np.asarray([image]))
            sys.stdout.write("\rProgress: %.2f%%" % (100.*count/100000.))
            sys.stdout.flush()

    print "\r"
    print 100.*cutcount/count
    
    #save arrays locally
    numpy_data_as_array=np.array(numpy_data)
    np.save('dataCos.npy',numpy_data_as_array)
    numpy_images_as_array=np.array(numpy_images)
    np.save('imagesCos.npy',numpy_images_as_array)

    
    
