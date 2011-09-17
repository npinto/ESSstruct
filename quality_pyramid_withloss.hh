/********************************************************
 *                                                      *
 *  Efficient Subwindow Search (ESS) implemented in C++ *
 *  bound for sum of grid cells, e.g. spatial pyramid   *
 *  a grid is really just a collection of boxes with    *
 *  right way to access them and add up their scores    *
 *                                                      *
 *   Copyright 2006-2008 Christoph Lampert              *
 *   Contact: <mail@christoph-lampert.org>              *
 *                                                      *
 *  Licensed under the Apache License, Version 2.0 (the *
 *  "License"); you may not use this file except in     *
 *  compliance with the License. You may obtain a copy  *
 *  of the License at                                   *
 *                                                      *
 *     http://www.apache.org/licenses/LICENSE-2.0       *
 *                                                      *
 *  Unless required by applicable law or agreed to in   *
 *  writing, software distributed under the License is  * 
 *  distributed on an "AS IS" BASIS, WITHOUT WARRANTIES *
 *  OR CONDITIONS OF ANY KIND, either express or        *
 *  implied. See the License for the specific language  *
 *  governing permissions and limitations under the     *
 *  License.                                            *
 *                                                      *
 ********************************************************/

#ifndef _QUALITY_PYRAMID_WITHLOSS_H
#define _QUALITY_PYRAMID_WITHLOSS_H

#include <vector>

#include "ess.hh"
#include "quality_pyramid.hh"

typedef struct {
      int numlevels;
      double** weightptr;
      int num_gt_boxes;
      double* gt_boxes;
} PyramidWithLossParameters;

class PyramidQualityWithLossFunction : public PyramidQualityFunction {

    public:
        void setup(int argnumpoints, int argwidth, int argheight, 
                   double* argxpos, double* argypos, double* argclst, 
                   void* argdata);
        double upper_bound(const sstate* state) const;

    private:
        std::vector<Box> ground_truth_boxes;

        double loss_bound(const sstate* state) const;

};

#endif
