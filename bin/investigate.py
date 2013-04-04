
# general
import numpy as np

# other spims
from compare import choose_weapon
from ims import Source, Pattern
from utility import overlaps, Match

def look_into_windows(si, pi, windows):
    """Do full-res last ditch comparisons.

    Compare the pattern and source at the best scalings we could find at 
    low res. 

    ----------
    si, pi : Image
       Source and Pattern to be compared.

    windows : dict
       Dictionary of scalings and windows that were found to be probable.

    ----------
    out : list[match]
       List of matches found by the comparisons. 
    """
    goodOnes = []
    for x in sorted(windows.keys(), key=lambda key: key[0]*key[1], 
                    reverse=True):
        
        # create window on source image
        swindow = Source.peerWindow(si, windows[x])

        pScaled = Pattern.resize(pi, x)
        method = choose_weapon(pScaled)
        nn, thresh, mean = method(swindow, pScaled)
        if mean < 1:
            nn[nn > 1.015] = 0
            mY, mX = np.where(nn > thresh)
            goodOnes += pack_the_goods(nn, si, pScaled, mX, mY, x, windows[x])

    return [Match(m[4], m[3], m[1][0], m[1][1], m[0])
            for m in overlaps(goodOnes)]

def pack_the_goods(nn, s, p, mX, mY, scaling, window):
    """Turn raw data into more raw data. 

    TODO: Make this sort of thing a class. 

    Packages data about matches into a format useful for the
    overlaps function.

    ----------
    nn : ndarray[float]
       Confidence matrix for the comparison at hand. 

    s, p : Image
       Images compared.

    mX, mY : list[int]
       Lists of x and y coordinates for potential matches. 

    scaling : tuple[int]
       Scaling of the Pattern. 

    window : tuple[int]
       A tuple illustrating the window from whence the measurement
       came.

    ----------
    out : list[...]
       List of packaged information about the potential matches without
       any overlapping matches.
    """
    vals = []
    for i,xi in enumerate(mX):
        yi = mY[i]
        c = nn[yi, xi]
        vals.append([c, [xi+window[0], yi+window[2]], scaling, p, s])
    if len(vals) > 0:
        vals.sort(key=lambda x: x[0], reverse=True)
    return vals
