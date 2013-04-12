
# general
import numpy as np

# specific
from fractions import Fraction

# other spims
from ims import Source, Pattern
from compare import choose_weapon

SLOP = 4.5

def possible_scales(s, p, fact):
    """Generate list of possible scalings.

    Finds all possible scalings of the pattern image that preserve 
    its aspect ratio and which allow it to fit entirely inside the
    source image. 

    ----------
    s : Source
       Source object which the scalings of the pattern image must
       fit within.

    p : Pattern
       Pattern object whose aspect ratio must be preserved.

    fact : int
       Coarse scale down factor used on the Pattern image.

    ----------
    out : dict
       Dictionary mapping each low-res scaling to one or more
       high-res scalings.
    """
    frac = Fraction(p.arr.shape[0],p.arr.shape[1])
    
    scales = {}

    currScaHigh = (p.arr.shape[0], p.arr.shape[1])
    currScaLow = (p.arr.shape[0]/fact, p.arr.shape[1]/fact)

    sX = int(s.arr.shape[0])
    sY = int(s.arr.shape[1])

    while(currScaHigh[0] <= sX and currScaHigh[1] <= sY):
        if currScaLow in scales.keys():
            scales[currScaLow].append(currScaHigh)
        else:
            scales[currScaLow] = [currScaHigh]
        
        currScaHigh = (currScaHigh[0]+frac.numerator, 
                       currScaHigh[1]+frac.denominator)
        currScaLow = (currScaHigh[0]/fact, currScaHigh[1]/fact)

    return scales

def res_factor(s, p):
    """Decides on a coarse scale down factor.

    Finds a feasible and (hopefully) clever coarse scale down factor 
    for the given Pattern and Source images. 

    ----------
    s : Source
       The larger image at risk in this scenario.

    p : Pattern
       The smaller image at risk in this scenario.

    ----------
    out : int
       Coarse scale down factor.
    """
    if p.arr.shape[0] > 1500 and p.arr.shape[1] > 1500:
        factor = 10
    elif p.arr.shape[0] > 10 and p.arr.shape[1] > 10:
        factor = 4
    else: 
        factor = 1
    return factor

def super_chunk_train_choo_choo(s, p):
    """The main boss of all scaling-comparison-related activities.

    Utilizes a sort of chunk search approach to finding potentially
    scaled subimages. Though the set is not ordered, it does peak
    near true scalings enabling us to search for those peaks using
    an iterative deepening/chunk search-type logic. 

    ----------
    s, p : Images
       The images to be compared.

    ----------
    out1 : dict
       The scalings of possible matches associated with the rough
       region the match was strongest in
       
    out2 : dict
       Associating low-res scalings with their (perhaps multiple) 
       high res counterparts
    """
    fact = res_factor(s, p)
    scales = possible_scales(s, p, fact)

    sd = Source.sizeDown(s, fact)
    pd = Pattern.sizeDown(p, fact)
    
    allLowRes = scales.keys()
    allLowRes.sort(reverse=True)
    n = len(scales) 
    
    step = max(n/100, 1)
    scatu = allLowRes

    while (step > 0 and len(scatu)/step > 0):
        scatu = scatu[::step]

        windows, strengths = train_cars(sd, pd, scatu, fact)

        if len(strengths) == 0:
            break; 
        
        best = max(strengths, key=lambda x: x[1])
        nextBest = [x[0]*step for x in strengths if x[1] > best[1] - .02]
        
        scatu = []
        for b in nextBest:
            begin = b - step; end = b + step
            if begin < 0:
                begin = 0
            if end > n:
                end = n
            scatu += allLowRes[begin:end]
        step = step/2 
        scatu = list(set(scatu))

    
    return real_windows(windows, scales)

def real_windows(wins, scals):
    """Convert from low-res windows to high-res.

    ----------
    wins : dict
       Associating low-res scalings to bounding windows.

    scales : dict
       Associating low-res scalings to high-res scalings.

    ----------
    out : dict
       Associating high-res scalings to bounding windows.
    """
    realWindows = {}
    for sca in wins.keys():
        for x in scals[sca]:
            realWindows[x] = wins[sca]

    return realWindows
    
def train_cars(s, p, scatu, fact):
    """Loop over and compare scalings given.
    
    Does the comparison at each level of scaling and records 
    the max value and probable window of promising comparisons.

    ----------
    s, p : Image
       Images to be compared. 

    scatu : list[tuple[int]]
       List of scalings to try.

    fact : int
       Factor by which both Images were scaled down.

    ----------
    out1 : dict
       Scalings associated with best-bet window in the Pattern
       for a match.
    out2 : list[tuple[int x float]]
       Tuple of scaling index in scatu and best match value for
       that comparison. 

    """
    windows = {}
    maxes = []

    for i,scaling in enumerate(scatu):

        pScaled = Pattern.resize(p, scaling)
                
        if abs(pScaled.arr.std() - p.stdev) > SLOP:
            break;

        # choose comparison method based on pScaled
        method = choose_weapon(pScaled)
        nn, thresh, mean = method(s, pScaled, fact)
        
        # post process and get the max
        nn[nn >= 1.015] = 0
        m = nn.max()
        
        mY, mX = np.where(nn >= thresh)

        if mX.size > 0: 
            windows[scaling] = train_caboose(mY, mX, fact, scaling)
            maxes += [(i, m)]
        
    return windows, maxes         
    

def train_caboose(mY, mX, fact, scaling):
    """Post-processing on good looking honkies.

    Finds full-scale window of potential matches in Source.

    ----------
    mY, mX : array[int]
       x and y coordinates of matches above threshold in 
       the probability matrix.
       
    fact : int
       Factor by which both Pattern and Source were scaled.

    scaling : tuple[int]
       The current scaling of the Pattern.

    ----------
    out : tuple[int]
       Length four tuple with corners of best match window.

    """
    mY.setflags(write=True); mX.setflags(write=True)
    mY *= fact; mX *= fact
    fullScaling = (scaling[0]*fact, scaling[1]*fact)

    return find_window(mX, mY, fullScaling)
    
    
def find_window(mX, mY, scaling, SL=10):
    """Constructs window encapsulating all given potential matches.

    ----------
    mX, mY : array[int]
       x and y coordinates of possible matches

    scaling : int
       The scaling of the Pattern whence the points came.

    SL : int, optional
       Slop level for the rectangle. Default is 10. 

    ----------
    out : tuple[int]
       (left,right,up,down) bounds of a box containing all
       points given with some level of slop. 

    """
    vals = []
    leftBound = mX[0]
    rightBound = mX[0]+scaling[1]+1
    upBound = mY[0]
    downBound = mY[0]+scaling[0]+1
    for i,xi in enumerate(mX):
        yi = mY[i]

        leftBound = min(leftBound, xi)
        rightBound = max(rightBound, xi+scaling[1]+1)
        upBound = min(upBound, yi)
        downBound = max(downBound, yi+scaling[0]+1)

    return (max(leftBound-SL, 0), rightBound+SL, max(upBound-SL,0), downBound+SL)
