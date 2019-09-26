from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

def getDataImage( path):
    #Read image
    img = Image.open( path )
    # create the pixel map
    pixels = img.load() 
    data = []
    pixel = []
    for i in range( img.size[0]):        # for every col do: img.size[0]
        for j in range( img.size[1] ):    # for every row   img.size[1]      
             pixel = pixels[i,j]          # get every pixel
             data.append( pixel[0] )
             data.append( pixel[1] )
             data.append( pixel[2] )

    #Viewing EXIF data embedded in image
    exif_data = img._getexif()
    exif_data
    return data

data1 = getDataImage('6c.png')
data2 = getDataImage('6d.png')
size = len( data1 )
net = buildNetwork( size, 12, 12, 4 )
ds = SupervisedDataSet( len( data1 ), 4)
ds.addSample( data1,(6, 6, 6, 6) )
ds.addSample( data2,(6, 6, 6, 6) )

trainer = BackpropTrainer(net, ds)
error = 10
iteration = 0
while error > 0.001: 
    error = trainer.train()
    iteration += 1
    print ( iteration, error )

print ("\nResult: ", net.activate( getDataImage('6ct.png') ))
print ("\nResult: ", net.activate( data2 ))

# Result: 2,035,868 0.12753658725156106
# Result: 2,604,332 0.127530177993413

