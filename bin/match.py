
# specific
from time import time
# other spims
from scale import super_chunk_train_choo_choo
from investigate import look_into_windows, one_shot_one_match
from ims import Pattern, Source

def match_master_ten_thousand(s, p, scaling=False):
    """Main matching engine.
    
    Loops over each pattern for each source image and runs subimage matching 
    routines on them.

    s, p : list
       list of strings referencing source and pattern images to be converted
       into Source and Pattern objects as needed.
    """
    p = [Pattern(pi) for pi in p]
    rm = []
    times = []
    for si in s:
        si = Source(si)
        for pi in p:

            t0 = time() # keep track of time for diagnostics

            if (si.arr.shape[0] >= pi.arr.shape[0] 
                and si.arr.shape[1] >= pi.arr.shape[1]):
                if scaling:
                    windows = super_chunk_train_choo_choo(si, pi)
                    rm += look_into_windows(si, pi, windows)
                else:
                    rm += one_shot_one_match(si, pi)

            times.append(time() - t0)
            
    return rm, times
