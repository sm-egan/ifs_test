# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:57:41 2020

@author: shann
"""

import numpy as np
import matplotlib.pyplot as plt
import math

class affineTransform:
    def __init__(self, in_array):
        self._Wmat = in_array[0:4].reshape((2,2))
        self._Wvec = in_array[4:6].reshape((2,1))
        self._det = np.linalg.det(self._Wmat)
        
    def transform(self, vec2d):
        return self._Wmat.dot(vec2d) + self._Wvec
    
class IFS:  
    #Constructor which uses existing affineTransform objects to construct the array of transformations
    # aTs can either be a single affineTransform, a list or array of affineTransforms
    def __init__(self, aTarr, nWs=1):
        self._nWs = nWs
        self._aTs = []
        if isinstance(aTarr, list) or type(aTarr).__module__ == np.__name__:
            #print("isinstance said True")
            for i in range(0, len(aTarr)):
                print(aTarr[i]._Wmat)
                self._aTs.append(aTarr[i])
        else:
            #print("isinstance said False")
            self._aTs.append(aTarr)
            
        self._weights = np.zeros(nWs)
        
    def set_weights(self, weightArr):
        self._weights = weightArr
        total = np.sum(self._weights) 
        if total  != 1.0:
            self._weights = self._weights/np.sum(self._weights)
    
    # Add just one affineTransform to the IFS
    def add_aT(self, aT):
        self._aTs.append(aT)
        self.nWs += 1
        
        #self.calculate_weights()
        
    def calculate_weights(self):
        self.weights = np.zeros(self.nWs)
        for i in range(0, self.nWs):
            self._weights[i] = np.linalg.det(self._aTs[i].Wmat)
        
        self._weights = self._weights/np.sum(np.abs(self._weights))
        
    def do_transform(self, vec2d):
        draw = np.random.uniform()
        #print("do_transform called. draw = " + str(draw))
        i=0
        while draw > np.sum(self._weights[0:i+1]) and i < len(self._aTs):
            i += 1
        #print(i)
        return self._aTs[i].transform(vec2d)
            
        
if __name__ == "__main__":
    
    image = "tree"
    aTarr = np.array([])
    weights = None
    if image == "fern":
        print('Setting affineTransformation for fern')
    # Defining affind transformations for a fern
        aTparams = np.array([[0,0,0,0.16,0,0],
                             [0.2,-0.26,0.23,0.22,0,1.6],
                             [-0.15,0.28,0.26,0.24,0,0.44],
                             [0.85,0.4,-0.4,0.85,0,1.6]
                             ])

        weights = np.array([0.01, 0.07, 0.07, 0.85])
        
    elif image == "tree":
        # Original tree IFS
        aTparams = np.array([[0,0,0,0.5,0,0],
                             [0.1,0,0,0.1,0,0.2],
                             [0.42,-0.42,0.42,0.42,0,0.2],
                             [0.42,0.42,-0.42,0.42,0,0.2]
                             ])
        '''
        # Modified IFS
        aTparams = np.array([[0.5,0,0,0.5,0,0],
                             [0.1,0,0,0.1,0,0.2],
                             [0.42,-0.42,0.42,0.42,0,0.2],
                             [0.42,0.42,-0.42,0.42,0,0.2]
                             ])
        '''
        

        weights = np.array([0.05, 0.15, 0.4, 0.4])
        
    elif image == "sierpinski":
        aTparams = np.array([[0.5,0,0,0.5,0,0],
                             [0.5,0,0,0.5,1,0],
                             [0.5,0,0,0.5,0.5,0.5],
                             ])
        
        weights = np.array([0.33,0.33,0.34])
        
    elif image == "spiral":
        phi = (1 + np.sqrt(5)) / 2
        r = 1/(6*phi)
        t = math.pi/12
        
        aTparams = np.array([[r*math.cos(t), r*math.sin(t), r*math.sin(t), r*math.cos(t), -1/phi, 1]])
        weights = np.array([1.0])
       
    # Construct an array of aTs for input into the IFS constructor
    for row in aTparams:
            aTarr = np.append(aTarr, np.array([affineTransform(row)]))
    ifs = IFS(aTarr, len(aTarr))
    ifs.set_weights(weights)
    
    xarray = [0]
    yarray = [0]
    
    point = np.array([[0],[0]])
    
    nit = 1000000
    ms = 0.5
    for n in range(0, nit):
        point = ifs.do_transform(point)
        xarray.append(point[0][0])
        yarray.append(point[1][0])
        
    fig1, ax1 = plt.subplots(figsize=(8,8))
    ax1.plot(xarray, yarray, 'o', color='g', markersize=ms)
    fig1.savefig(image + '-' + str(int(nit / 1000)) + 'k_it-ms' + str(ms) + '.png')
    
        
    