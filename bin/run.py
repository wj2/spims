
# general
import numpy as np
import sys, os

# specific
from PIL import Image
from time import time

# other spims
from match import match_master_ten_thousand

def get_input_list(s):
    """Make list of images depending on input.
    
    s : String
       file or folder to be made into list
    """
    if os.path.isdir(s):
        return [s+x for x in os.listdir(s)]
    else:
        return [s]            

def main(patterns, sources, printMatches=True, diag=False):
    """Main program function.

    Do subimage matching for given inputs and print (or not)
    the result.

    patterns, sources : str
       Represent input images to be compared

    printMatches : Boolean, optional
       if True, print matches to stdout; else, return them

    diag : Boolean, optional
       if True, print or return diagnostic information; else
       do nothing of the sort
    """
    t0 = time()

    patterns = get_input_list(patterns)
    sources = get_input_list(sources)

    # comparisons
    matches, diagd = match_master_ten_thousand(sources, patterns)

    if printMatches == True:
        for m in matches:
            sys.stdout.write(m.toString())

        if diag:
            print '\nTotal time: ' + str(time() - t0)
            print ('Compared ' + str(len(diagd)) + ' images with an average \n'
                   'time of '+str(round(np.mean(diagd),3))+'s per comparison \n'
                   'and a std of '+str(round(np.std(diagd),3))+'s.')
    else:
        diagd = {"total_time":str(time() - t0),
                "comp_num":str(len(diagd)),
                "avg_time":str(round(np.mean(diagd),3)),
                "std_time":str(round(np.std(diagd),3))}
        return matches, diagd

def parse_opts(opt):
    """Nasty options parser.
    
    Does nasty things to parse the options because we didn't think of what
    that other group did.

    opt : list
       list of options from command line
    """
        
    if len(opt) != 4:
        sys.stderr.write('IOError: Malformed input\n')
        sys.exit(1)
    opt = [(opt[0],opt[1]),(opt[2],opt[3])]
    if opt[0][0] == '-p':
        if os.path.isdir(opt[0][1]):
            sys.stderr.write('IOError: Expected pattern file\n')
            sys.exit(1)
        else: pattern = opt[0][1]
    elif opt[0][0] == '-pdir' or opt[0][0] == '--pdir':
        if os.path.isdir(opt[0][1]):
            pattern = opt[0][1] + '/'
        else: 
            sys.stderr.write('IOError: Expected pattern dir\n')
            sys.exit(1)
    if opt[1][0] == '-s':
        if os.path.isdir(opt[1][1]):
            sys.stderr.write('IOError: Expected source file\n')
            sys.exit(1)
        else: source = opt[1][1]
    elif opt[1][0] == '-sdir' or opt[1][0] == '--sdir':
        if os.path.isdir(opt[1][1]):
            source = opt[1][1] + '/'
        else: 
            sys.stderr.write('IOError: Expected source dir\n')
            sys.exit(1)
    else:
        sys.stderr.write('IOError: Malformed input\n')
        sys.exit(1)

    return pattern, source
