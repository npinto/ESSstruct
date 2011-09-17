 ********************************************************
 *                                                      *
 * Structured Regression Training with SVMpython/struct *
 *                                                      *
 * version 1.1, August 4th 2008                         *
 *                                                      *
 *   Copyright 2006-2008 Christoph Lampert              *
 *   Contact: <mail@christoph-lampert.org>              *
 *                                                      *
 *  We perform structured regression training to learn  *
 *  the best weights for sliding window-like object     *
 *  localization, see                                   *
 *       Matthew B. Blaschko, Christoph H. Lampert:     *
 *       Learning to Localization Objects With          *
 *       Structured Output Regression, ECCV (2008)      *
 *                                                      *
 ********************************************************

This is a submodule for the SVMpython wrapper by T. Finley
http://www.cs.cornell.edu/~tomf/svmpython2/
to SVMstruct by T. Joachims:
http://svmlight.joachims.org/svm_struct.html

It can be used to train a structured regression SVM 
for object localization, see our ECCV2008 paper:
Matthew B. Blaschko, Christoph H. Lampert: "Learning 
to Localization Objects With Structured Output Regression".

Note that this code is meant as illustration of how to use 
structured regression for object localization. It is not 
a complete system for object localization. 

===========================================================


subwindow.py requires the ESS library to be loadable from 
Python, please adapt the path in lines 33 and/or 35. 

It currently relies on two kinds of input data that is 
adapted to the VOC challenges:

1) a file of ground truth boxes (one per line) in the format

ID   left top right bottom

where ID is the 6digit numeric image ID 
and [left,top,right,bottom] is a ground truth box instance.

For images without boxes (negative training example) use

ID

without coordinates. However, negative examples are not 
necessary for training and not always beneficial. 

2) For each image (ID), subwindow.py requires a file '%06d.clst' 
in the format

x y clst
...

where each (x,y) is the coordinate of a feature point and 'clst'
is its corresponding cluster ID in a bag-of-visual-words 
histogram (starting at 0). 

The resulting model-file contains a 'w' vector that can be
used for regular sliding window or efficient subwindow search 
(ESS) localization.


===========================================================

EXAMPLE:

./svm_python_learn --m subwindow -c 0.001 car.gtbox car.model

OUTPUT:

Reading training examples...1 examples read
done
size_psi set to 3000
Setting current working precision to 50.
Iter 1 (1 active): +*----------------------------------------------------------
(NumConst=1, SV=1, CEps=1.7044, QPEps=0.0000)
Setting current working precision to 25.
Iter 2 (1 active): +*----------------------------------------------------------
(NumConst=1, SV=1, CEps=1.7044, QPEps=0.0000)
Setting current working precision to 12.5.
Iter 3 (1 active): +*----------------------------------------------------------
(NumConst=1, SV=1, CEps=1.7044, QPEps=0.0000)
Setting current working precision to 6.25.
Iter 4 (1 active): +*----------------------------------------------------------
(NumConst=1, SV=1, CEps=1.7044, QPEps=0.0000)
Setting current working precision to 3.125.
Iter 5 (1 active): +*----------------------------------------------------------
(NumConst=1, SV=1, CEps=1.7044, QPEps=0.0000)
Setting current working precision to 1.5625.
Iter 6 (1 active): .*----------------------------------------------------------
(NumConst=2, SV=2, CEps=1.7044, QPEps=0.0000)
Iter 7 (1 active): +(NumConst=2, SV=2, CEps=0.6891, QPEps=0.0000)
Setting current working precision to 0.78125.
Iter 8 (1 active): +*----------------------------------------------------------
(NumConst=2, SV=2, CEps=0.6891, QPEps=0.0000)
Setting current working precision to 0.390625.
Iter 9 (1 active): .*----------------------------------------------------------
(NumConst=3, SV=2, CEps=0.6891, QPEps=0.0000)
Iter 10 (1 active): .*----------------------------------------------------------
(NumConst=4, SV=3, CEps=0.4445, QPEps=0.0000)
Iter 11 (1 active): +(NumConst=4, SV=3, CEps=0.2772, QPEps=0.0000)
Setting current working precision to 0.195312.
Iter 12 (1 active): .*----------------------------------------------------------
(NumConst=5, SV=4, CEps=0.2772, QPEps=0.0000)
Iter 13 (1 active): .*----------------------------------------------------------
(NumConst=6, SV=5, CEps=0.3330, QPEps=0.0000)
Iter 14 (1 active): .*----------------------------------------------------------
(NumConst=7, SV=5, CEps=0.2791, QPEps=0.0000)
Iter 15 (1 active): .*----------------------------------------------------------
(NumConst=8, SV=6, CEps=0.2351, QPEps=0.0000)
Iter 16 (1 active): +(NumConst=8, SV=6, CEps=0.1532, QPEps=0.0000)
Setting current working precision to 0.1.
Iter 17 (1 active): .*----------------------------------------------------------
(NumConst=9, SV=6, CEps=0.1532, QPEps=0.0000)
Iter 18 (1 active): .*----------------------------------------------------------
(NumConst=10, SV=7, CEps=0.1452, QPEps=0.0000)
Iter 19 (1 active): .*----------------------------------------------------------
(NumConst=11, SV=8, CEps=0.1534, QPEps=0.0000)
Iter 20 (1 active): .*----------------------------------------------------------
(NumConst=12, SV=7, CEps=0.1634, QPEps=0.0000)
Iter 21 (1 active): .*----------------------------------------------------------
(NumConst=13, SV=7, CEps=0.1128, QPEps=0.0000)
Iter 22 (1 active): +(NumConst=13, SV=7, CEps=0.0999, QPEps=0.0000)
Final epsilon on KKT-Conditions: 0.10000
Upper bound on duality gap: 0.00010
Dual objective value: dval=0.00005
Total number of constraints in final working set: 13 (of 12)
Number of iterations: 22
Number of calls to 'find_most_violated_constraint': 22
Number of SV: 7 
Number of non-zero slack variables: 0 (out of 1)
Norm of weight vector: |w|=0.00988
Norm. sum of slack variables (on working set): sum(xi_i)/n=0.00000
Norm of longest difference vector: ||Psi(x,y)-Psi(x,ybar)||=752.16621
Runtime in cpu-seconds: 217.58 (0.03% for QP, 92.99% for Argmax, 6.97% for Psi, 0.00% for init)
Writing learned model...done

