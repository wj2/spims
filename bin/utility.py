
# general
import numpy as np
import sys, os

#specific
from PIL import Image
from time import time


FILETYPES = ['GIF','JPEG','PNG']

def imread(fname):
    """Read image from file. 

    Nearly verbatim of scipy.imread, but has special provision
    to restrict input types. 

    ----------
    fname : str
       Path to file to be read. 

    ----------
    out1 : ndarray[float] 
       Greyscale 2d array representation of the image.

    out2 : ndarray[float]
       RGG 3 depth 2d array representation of the image. 

    """
    try:
        fp = open(fname, 'rb')
        im = Image.open(fp)
    except:
        sys.stderr.write('IOException: Invalid input type on '+fname+'\n')
        sys.exit(1)
    else:
        if im.format not in FILETYPES:
            sys.stderr.write('IOException: Invalid image type\n')
            sys.exit(1)
            
    fa = np.array(im.convert('F'))
    im = im.convert('RGB')
    wa = np.array(im)
    
    fp.close()

    return fa, wa

def pad_zeroes(m, axis, thick):
    """Pads given matrix with zeros. 

    Pads m with thick zeros on axis. 

    ----------
    m : ndarray
       Matrix to be padded.

    axis : int
       Axis of m to be padded.

    thick : int
       Number of zeros to pad with.

    ----------
    out1 : ndarray
       Simply m, padded with given number of zeros on given axis.
    """

    if axis == 1:
        rows = m.shape[0]
        cols = thick
    elif axis == 0:
        rows = thick
        cols = m.shape[1]

    return np.concatenate((m, np.zeros((rows,cols))), 
                          axis)

# pad a to size of b, with zeroes
def pad_to_size_of(a, b):
    """Pads first matrix to the size of second matrix.

    Uses zeros to pad first matrix to size of second matrix.
    Will fail if first matrix is larger than second. 

    ----------
    a, b : ndarray

    ----------
    out : ndarray
       Original first array padded with zeros s.t it is
       now the size of b. 
    """
    a2 = pad_zeroes(a, 0, b.shape[0] - a.shape[0])
    return pad_zeroes(a2, 1, b.shape[1] - a.shape[1])

def overlaps(vals, perc=.5):
    """Eliminates overlapping vals.

    Examines vals based on their size and position to choose
    the strongest vals and eliminate weaker vals that overlap
    with them. 

    ----------
    vals : list[float x list[int x int] 
                x tuple[int x int] x Pattern 
                x Source]
       Representation of a match in the larger context of the project.

    perc : float, optional
       Percent overlap at which to throw out a worser match

    ----------
    out : list[...]
       Same list as before with any overlapping matches removed. 
    """
    underlaps = []
    if len(vals) > 0:
        underlaps.append(vals[0])
        for val in vals:
            i = 0
            amount = len(underlaps)
            inv = 1
            while (inv):

                if (i == amount):
                    underlaps.append(val)
                    inv = False
                elif poverlap(underlaps[i][1], val[1],
                              underlaps[i][2], val[2]) >= perc:
                    if val[0] > underlaps[i][0]:
                        underlaps[i] = val
                    inv = False
                else:
                    i += 1
  
    return underlaps

def poverlap(t1, t2, size1, size2):
    """Calculate percent overlap.
    
    Calculates percent overlap of two given points with their
    given sizes. 

    ----------
    t1, t2 : tuple[int]
       Tuples of coordinates of the match top-left corner.

    size1, size2 : tuple[int]
       Tuples of sizes of both t1 and t2, respectively. 

    ----------
    out : float
       Percent overlap of t1 and t2.
    """
    x0 = t1[0]
    y0 = t1[1]
    x1 = t1[0] + size1[0]
    y1 = t1[1] + size1[1]

    x2 = t2[0]
    y2 = t2[1]
    x3 = t2[0] + size2[0]
    y3 = t2[1] + size2[1]
    
    ol = max(0, min(x1, x3) - max(x0, x2)) * max(0, min(y1, y3) - max(y0, y2))

    return ol / float(2*(size2[0]*size2[1]) - ol)

class Match:

    def __init__(self, source, pattern, x, y, prob):
        self.source = source.name
        self.pattern = pattern.name
        self.prob = prob 
        self.w = pattern.arr.shape[1]
        self.h = pattern.arr.shape[0]
        self.x = x
        self.y = y

    def __str__(self):
        return (self.pattern + " matches " + self.source
                + " at " + str(self.w) + "x" + str(self.h) + 
                "+" + str(self.x) + "+" + str(self.y))+"\n"

    def toStringTest(self):
        return (self.pattern + " matches " + self.source 
                + " at " + str(self.w) + "x" + str(self.h) 
                + "+" + str(self.x) + "+" + str(self.y) 
                + " with " + str(round(self.prob, 2)) + " confidence\n")
    
