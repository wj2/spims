
# specific
from numpy.fft import fftn
from scipy.misc import imresize
from copy import deepcopy

# other spims
from utility import imread

class Im:
    def __init__(self, path):
        self.path = path
        self.name = path.split('/')[-1]
        self.arr, self.warr = imread(path)
        self.stdev = self.arr.std()
        self.mean = self.arr.mean()

    @staticmethod
    def sizeDown(im, fact):
        im2 = deepcopy(im)
        im2.arr = im2.arr[::fact,::fact]
        im2.warr = im2.warr[::fact,::fact,:]
        im2.stdev = im2.arr.std()
        im2.mean = im2.arr.mean()
        return im2

    @staticmethod
    def resize(im, scaling):
        im2 = deepcopy(im)
        if im2.arr.shape == tuple(scaling):
            return im2
        im2.arr = imresize(im2.arr, scaling)
        im2.warr = imresize(im2.warr, scaling)
        im2.stdev = im2.arr.std()
        im2.mean = im2.arr.mean()
        return im2

class Source(Im):
    
    def __init__(self, path):
        Im.__init__(self,path)
        self.fft = fftn(self.arr)
        self.fft2 = fftn(self.arr ** 2)

    @staticmethod
    def sizeDown(im, fact):
        im2 = Im.sizeDown(im, fact)
        im2.fft = fftn(im2.arr)
        im2.fft2 = fftn(im2.arr ** 2)
        return im2

    @staticmethod
    def peerWindow(im, bounds):
        im2 = deepcopy(im)
        
        im2.arr = im2.arr[bounds[2]:bounds[3],
                          bounds[0]:bounds[1]]
        im2.warr = im2.warr[bounds[2]:bounds[3],
                            bounds[0]:bounds[1], :]
        im2.fft = fftn(im2.arr)
        im2.fft2 = fftn(im2.arr ** 2)
        im2.stdev = im2.arr.std()
        im2.mean = im2.arr.mean()

        return im2


class Pattern(Im):
    
    pass
