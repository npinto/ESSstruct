"""A module for SVM^python for object localization."""

# Christoph Lampert, chl@tuebingen.mpg.de, 2007-2008
#
# for descriptions of the learning see
# [1] Blaschko, Lampert: "Object Localization by Structured Regression", ECCV 2008
# for the branch-and-bound method to find the most violated constraint:
# [2] Lampert, Blaschko, Hoffman: "Beyond Sliding Windows: ..." CVPR 2008

import sys
import svmapi

# prepare numeric stuff for branch-and-bound search
from ctypes import c_int, c_double, Structure, byref
import pylab
import numpy
from numpy import array,ones,histogram,hstack
from numpy.ctypeslib import load_library,ndpointer
import numpy
numpy.seterr("raise") 

class Box_struct(Structure):
        """Structure to hold left,top,right,bottom and score of a box instance.
           The fields have to coincide with the C-version in pyramid_search.h
        """
        _fields_ = [("left", c_int), 
                    ("top", c_int), 
                    ("right", c_int), 
                    ("bottom", c_int), 
                    ("score", c_double) ]

try:
    pyramidlib = load_library("libess.so","/kyb/agbs/chl/libs/")
except OSError:
    pyramidlib = load_library("libess.so","/agbs/cluster/chl/libs/")

pyramidlib.pyramid_search_structured.restype = Box_struct
pyramidlib.pyramid_search_structured.argtypes = [c_int,c_int,c_int,     # numpoints, width, heigth,
        ndpointer(dtype=c_double, ndim=1, flags='C_CONTIGUOUS'),        # argxpos
        ndpointer(dtype=c_double, ndim=1, flags='C_CONTIGUOUS'),        # argypos
        ndpointer(dtype=c_double, ndim=1, flags='C_CONTIGUOUS'),        # argclst
        c_int, c_int,                                                   # numclusters, numlevels
        ndpointer(dtype=c_double, ndim=1, flags='C_CONTIGUOUS'),        # 1D weight vector
        c_int, ndpointer(dtype=c_double, ndim=1, flags='C_CONTIGUOUS')] # ground truth size, ground truth box

class training_image:
    def __init__(self, imageid, xyc):
        self.id = imageid
        self.x = array([int(val) for val in xyc[:,0]], c_double)
        self.y = array([int(val) for val in xyc[:,1]], c_double)
        self.c = array([int(val) for val in xyc[:,2]], c_double)
        self.n = xyc.shape[0]
        self.width = int(max(self.x))+1
        self.height = int(max(self.y))+1

def array_to_Box(coords=[0,0,0,0], score=0.):
        return Box_struct(coords[0],coords[1],coords[2],coords[3],score)

def Box_to_array(box):
        box_values = []
        if isinstance(box, list):
            for b in box:
                box_values.extend([b.left,b.top,b.right,b.bottom,b.score])
        else:
            box_values.extend([box.left,box.top,box.right,box.bottom,box.score])
        return array(box_values, c_double)

def parse_parameters(sparm):
    """Sets attributes of sparm based on command line arguments.
    
    This gives the user code a chance to change sparm based on the
    custom command line arguments.  The custom command line arguments
    are stored in sparm.argv as a list of strings.  The custom command
    lines are stored in '--option', then 'value' sequence.
    
    If this function is not implemented, any custom command line
    arguments are ignored and sparm remains unchanged."""
    sparm.numbins = 3000    # number of bag-of-visual-word codebooks
    sparm.numlevels = 1     # no spatial pyramid 


def read_examples(filename, sparm):
    """Parses an input file into an example sequence.
       Currently, we parse only raw-test-file in the VOC format:

       # ID   left top right bottom
       where [left,top,right,bottom] is one box instance, or just
       # ID
       if there is no box.
       
       e.g. 
       005181 224 255 246 276
       005181 90 238 112 258
       005194 151 88 455 418
       005259 72 265 124 298
       005259 124 259 166 298
       005259 165 267 191 294
       000002
       000003
       000005
   
       Note that duplicate IDs are allowed for the positive class.
       Those are combined into single examples with multiple labels,
       should can be handle by the delta-loss.
       
       We read only the IDs and the feature files themselves, since 
       they are only needed in the branch-and-bound C-routine anyway.
   """
    QUANT_FACTOR=1 # quantize all coordinated by this (extra speedup)
    
    # This routine reads box files of the VOC type
    examples = []
    lastid = -1

    # Open the file and read each example.
    path = 'train/'
    for line in file(filename):
        # Get rid of comments.
        if line.find('#'): 
            line = line[:line.find('#')]
        tokens = line.split()
        # If the line is empty, who cares?
        if not tokens: 
            continue

        # Get the ID (goes into features) and box (goes into target)
        imageid = int(tokens[0])
        if (imageid != lastid):
            try: 
                xyc = pylab.load(path + '%06d.clst' % imageid)
                xyc[:,0] /= QUANT_FACTOR
                xyc[:,1] /= QUANT_FACTOR
            except IOError:
                path = 'test/'
                xyc = pylab.load(path + '%06d.clst' % imageid)
                xyc[:,0] /= QUANT_FACTOR
                xyc[:,1] /= QUANT_FACTOR
    
        examples.append( [training_image(imageid, xyc), []] ) # automatic xyc split, no label(s)
        lastid = imageid

        target = [int(x)/QUANT_FACTOR for x in tokens[1:]]
        if (len(target) == 4):
            target=array_to_Box(target,1.)      # we use the score as indicator whether it's a positive box
            examples[-1][1].append(target)
        else:
            width=examples[-1][0].width
            height=examples[-1][0].height
            target=array_to_Box([0,0,0,0],-1.)  # '-1' means it's a negative full image
            examples[-1][1].append(target)
            
    # Print out some useful statistics.
    print len(examples), 'examples read'
    return examples


def init_model(sample, sm, sparm):
    """Initialize the size of bins for the weight vector."""
    # 
    sm.size_psi = sparm.numbins   # for a spatial pyramid, size_psi is total histogram length
    print 'size_psi set to',sm.size_psi


def classification_score(ex,y,sm,sparm):
    """Return score for (example, label) pair. Unclear if this is in active use"""
    # Utilize the svmapi.Model convenience method 'classify'. 
    # 
    return sm.svm_model.classify(psi(ex,y,sm,sparm))


def classify_example(ex, sm, sparm):
    """Returns the classification of an example 'x'. This routine is only used during testing."""

    # We should make this a command-line option, or determine it from somewhere.
    # We have to set this 'again', because this part is used in testing, the other in training.
    sparm.numbins=3000
    sparm.numlevels=1
    best_box = find_most_violated_constraint(ex, [], sm, sparm)

    # For the VOC ranking task, returning a box with bad score is never worse than not 
    # returning it. If the aim to classification, we might want to return 'no box' if the
    # score is 0 or less.
    return best_box


def find_most_violated_constraint(ex,y,sm,sparm):
    """Returns the most violated constraint for example (ex,box)."""
    # This is the crucial routine for the max-margin training with a joint kenel map. 
    # For object localization, we can't do the search exhaustively. Instead, we 
    # call the efficient branch-and-bound search from [1].

    # w should be an array anyway, but making sure doesn't hurt
    w = array([val for val in sm.w], c_double)

    # convert the ground truth box-structures to a numpy array
    ground_truth_boxes = Box_to_array(y)
    num_ground_truth_boxes=0
    if len(ground_truth_boxes)>0:
        if ground_truth_boxes[4]>0:
            num_ground_truth_boxes = len(ground_truth_boxes)/5

    #print ground_truth_boxes
    # do the search itself.
    box = pyramidlib.pyramid_search_structured(ex.n, ex.width, ex.height, ex.x, ex.y, ex.c, \
                     sparm.numbins, sparm.numlevels, w, num_ground_truth_boxes, ground_truth_boxes)

    # Here's a good place to check if both the implementation of 'most_violated_constraint' coincide.
    #
    #assert( abs(classification_score(ex, box, sm, sparm)+loss(y,box,sparm)-box.score) < 0.0001 )
    #score1 = classification_score(ex, box, sm, sparm)
    #loss1 = loss(y,box,sparm)
    #if abs(score1+loss1-box.score) >= 0.0001:
    #  print "Warning! Classifcation score differs from box.score: %f+%f != %f" % (score1,loss1,box.score)
    #print >> sys.stderr, score1,loss1, score1+loss1, " : " ,
    #write_label(sys.stderr,box)
    return box


def psi(ex, y, sm, sparm):
    """Returns the combined feature vector Psi(x,y).
       For us this is the histogram of the image x restricted to the box y.
       y should derive from 'find_most_violated_costraint' and be a real box. 
       If it's a list, we use just the first box inside. If the list is empty,
       we use the whole image (used in testing).
    """

    if isinstance(y, list): # in theory, this should only be called with single boxes as y.
         y=y[0]             # but just in case, for a list of boxes we use only the first one
    
    # Crop the image ex to the region given by y
    idx = (ex.x >= y.left) & (ex.x <= y.right) & (ex.y >= y.top) & (ex.y <= y.bottom)
    clsthist = histogram(ex.c[idx], xrange(sparm.numbins))[0]
    return svmapi.Sparse(clsthist)


def init_constraints(sample, sm, sparm):
    """Initializes special constraints, if necessary. 

    The constraints "F(x, box ) - F(x,no box) > 1-xi" for all x with a 'real' box remain
    all through-out learning. We add them to the constraint set once in the beginning. 
    This also fixes some degenerate behaviour of SVMstruct when starting with w==0.
    """
    import svmapi
    constraints = []
    curslack = len(sample)+1
    for feature,boxlist in sample:
        for box in boxlist:
            if box.score < 0:
               continue # use images which contain an object instance

            feature_hist = psi(feature, box, sm, sparm)
            sparse = svmapi.Sparse( feature_hist )
            lhs = svmapi.Document([sparse], costfactor=1, slackid= curslack+100000)

            # Append the lhs and the rhs to the constraint set
            constraints.append((lhs, 1.))
    return constraints


# Boxes _include_ their boundaries, i.e.g even a box with left==right and top==bottom 
# still has size 1. Only illegal boxes (left>right) or (top>bottom) can have size 0 or less.
def box_area(box):
        return (box.right-box.left+1)*(box.bottom-box.top+1)

# Box overlap is box_area(intersection)/area(union). The 'union' is not a rectangle, but
# luckily we can calculate area(union) = box_area(box1)+box_area(box2)-box_area(intersection)
def box_overlap(box1, box2):
        intersection = array_to_Box([ max(box1.left,box2.left), max(box1.top,box2.top), \
                                      min(box1.right,box2.right), min(box1.bottom,box2.bottom) ])
        if (intersection.left<=intersection.right) & (intersection.top<=intersection.bottom):
                return box_area(intersection)/float(box_area(box1)+box_area(box2)-box_area(intersection))
        else:
                return 0.

def loss_function(overlap):
        return (1.-overlap)     # simplest choice
        #return max(1.-2*overlap,0.) # 50% overlap tolerance region 
        #if overlap>=0.5:    # 0-1 loss as VOC scores with 
        #   return 0
        #else:
        #   return 1

def loss(y, ybar, sparm):
    """Loss function for label pair (y,ybar).
       In training, y is a list of ground truth boxes. ybar is a single candiate box.
       For 'positive' examples, the loss is 1 minus the area overlap of the boxes. 
       For 'negative' examples, the loss is the box overlap with the full image.
       In testing, y is empty, and the loss is zero.
    """

    if (y == []): # a test time there are no ground truth boxes => Delta=0
       return 0.

    if (y[0].score < 0. ): # negative example: Delta(box, nobox)=1 (don't predict a box)
        return 1.

    # positive example: it's enough if the prediction overlaps _one_ of the ground truth boxes
    maxoverlap = max([ box_overlap(box, ybar) for box in y])
    return loss_function(maxoverlap)


def write_label(fileptr, y):
    """Write a predicted label to an open file.
       We use the usual format for VOC prediction task (except for the ID)
    """
    print>>fileptr, y.score, y.left, y.top, y.right, y.bottom


def print_iteration_stats(ceps, cached_constraint, sample, sm,
                          cset, alpha, sparm):
    """Called after each iteration. Currently, we don't do anything, but 
       we could e.g. use it to write the best model learned so far to disk."""

    #import cPickle, bz2
    #filename = 'temp.model' # unfortunately, we don't know a good filename 
    #cPickle.dump(sm, bz2.BZ2File(filename, 'w') )

