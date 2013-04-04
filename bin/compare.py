
# general
import numpy as np

# specific
from fractions import Fraction
from numpy.fft import fftn, ifftn, fft2, ifft2
from scipy.misc import imresize

# other spims
from utility import pad_to_size_of, overlaps
from ims import Source, Pattern

GEN_THRESH = .935
SMALL_THRESH = .999

def choose_weapon(p):
    """Entry method into the world of comparison!

    Takes a gander at the given Pattern and decides
    which weapon would be most suitable.

    ----------
    p : Pattern
    
    ----------
    out : function
       The comparison method determined to be most suitable.

    """
    if p.warr.size == 3:
        return brute_force_single
    elif p.arr.std() < 0.0001:
        return brute_force_std
    else:
        return nccfft

def single_arrow(s, p, fact=1):
    """Used for single pixel Pattern images.

    ----------
    s, p : Image
       Pattern and Source images for comparison.
       
    fact : int, optional
       Factor by which both Source and Pattern are
       scaled down.

    ----------
    out1 : ndarray[float]
       Confidence matrix for matches.

    out2 : float
       Threshold for deciding if a match has been found.

    out3 : float
       Mean of the confidence matrix.
    """
    
    notNorm = np.zeros(s.warr.shape)
    notNorm[s.warr == p.warr[0]] = 1
    nn = notNorm.mean(axis=2)
    
    return nn, SMALL_THRESH, nn.mean()

def barrage_of_arrows(s, p, fact=1):
    """Used for single, solid color Pattern images.

    ----------
    s, p : Image
       Pattern and Source images for comparison.
       
    fact : int, optional
       Factor by which both Source and Pattern are
       scaled down.

    ----------
    out1 : ndarray[float]
       Confidence matrix for matches.

    out2 : float
       Threshold for deciding if a match has been found.

    out3 : float
       Mean of the confidence matrix.
    """

    c = p.warr[0,0]
    nn = np.zeros(s.arr.shape)
    nn_temp = np.zeros(s.warr.shape)
    nn_temp[s.warr == c] = 1
    nn_temp = nn_temp.mean(axis=2)
    xs, y = np.where(nn_temp == 1)

    for i,xi in enumerate(xs):

        if (xi+p.warr.shape[0] <= s.warr.shape[0]
            and y[i]+p.warr.shape[1] <= s.warr.shape[1]):                
            con = (p.warr == s.warr[xi:xi+p.warr.shape[0],
                                    y[i]:y[i]+p.warr.shape[1]])
            if con.all():
                nn[xi,y[i]] = 1

    return nn, SMALL_THRESH, nn.mean()

def nccfft(s, p, fact=1):
    """Used for all Patterns that do not fall other categories.

    Cross correlates normalized Source and Pattern images while
    taking advantage of FFTs for the convolution step. 
    
    ----------
    s, p : Image
       Pattern and Source images for comparison.
       
    fact : int, optional
       Factor by which both Source and Pattern are
       scaled down.

    ----------
    out1 : ndarray[float]
       Confidence matrix for matches.

    out2 : float
       Threshold for deciding if a match has been found.

    out3 : float
       Mean of the confidence matrix.
    """
    
    # subtract mean from Pattern
    pmm = p.arr - p.arr.mean()
    pstd = p.arr.std()
    n = p.arr.size
    
    # make matrix of ones the same size as pattern
    u = np.ones(p.arr.shape)

    # pad matrices (necessary for convolution)
    upad = pad_to_size_of(u, s.arr)
    pmmpad = pad_to_size_of(pmm, s.arr)

    # compute neccessary ffts
    fftppad = fftn(pmmpad)
    ffts = s.fft
    fftss = s.fft2
    fftu = fftn(upad)

    # compute conjugates
    cfppad = np.conj(fftppad)
    cfu = np.conj(fftu)

    # do multiplications and ifft's
    top = ifftn(cfppad * ffts)
    bot1 = n * ifftn(cfu * fftss)
    bot2 = ifftn(cfu * ffts) ** 2

    # finish it off!
    bottom = pstd * np.sqrt(bot1 - bot2)
    full = top / bottom

    return nccfft_post(full, fact, p)

def nccfft_post(full, fact, p):
    """Post proccessing on the results from nccfft. 

    nccfft is so powerful that it needs a whole nother function to
    determine the appropriate threshold, crop out meaningless values, 
    and calculate its mean. 

    ----------
    full : ndarray[float]
       The full confidence matrix from nccfft.
       
    fact : int
       The scale factor applied to both Source and Pattern.

    p : Pattern
       The pattern image used in the comparison.

    ----------
    out1 : ndarray[float]
       Cropped confidence matrix for matches.

    out2 : float
       Threshold for deciding if a match has been found.

    out3 : float
       Mean of the confidence matrix.

    """
    probMatrix = full.real[:-p.arr.shape[0]+1,:-p.arr.shape[1]+1]
    pmstd = probMatrix.std()
    pmean = probMatrix.mean()
    if pmstd < .1:
        if fact > 1:
            thresh = min(pmean + .60, .70)
        else:
            thresh = .95
    else:
        if fact > 1:
            thresh = min(pmean + 5.0  * pmstd, .69)
        else:
            thresh = min(pmean + 5.5  * pmstd, .984)
            
    return probMatrix, thresh, pmean

