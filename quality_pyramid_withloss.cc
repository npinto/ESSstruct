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

/* quality_pyramid with loss uses the pyramid quality function,
   but it adds a loss function to the score.
   For structured regression, this is the Delta(y,y') term,
   but it could also be something else, e.g. a shape prior.
*/ 

#include <vector>
#include <cmath>

#include "ess.hh"
#include "quality_pyramid_withloss.hh"

void PyramidQualityWithLossFunction::setup(int argnumpoints, int argwidth, int argheight, 
                               double* argxpos, double* argypos, double* argclst, 
                               void* argdata) {
    PyramidWithLossParameters* data = reinterpret_cast<PyramidWithLossParameters*>(argdata);

    PyramidParameters tempdata = { data->numlevels, data->weightptr };
    PyramidQualityFunction::setup(argnumpoints, argwidth, argheight, argxpos, argypos, argclst, &tempdata);

    ground_truth_boxes.resize(data->num_gt_boxes);
    for (unsigned int i=0; i<data->num_gt_boxes; i++) {
        ground_truth_boxes[i].left = static_cast<int>(data->gt_boxes[5*i]);
        ground_truth_boxes[i].top = static_cast<int>(data->gt_boxes[5*i+1]);
        ground_truth_boxes[i].right = static_cast<int>(data->gt_boxes[5*i+2]);
        ground_truth_boxes[i].bottom = static_cast<int>(data->gt_boxes[5*i+3]);
        ground_truth_boxes[i].score = static_cast<int>(data->gt_boxes[5*i+4]);
    }
    return;
}

// boxes _include_ their coordinates, so e.g. the box [0,0,0,0] has size 1.
static inline double box_area(Box box) {
        return (box.right-box.left+1.)*(box.bottom-box.top+1.);
}

static double box_overlap_lower(const sstate* s, const Box ref_box) {
    const Box min_box = { s->high[0], s->high[1], s->low[2], s->low[3], 0. };
    if ((min_box.right < min_box.left) || (min_box.bottom < min_box.top))
        return 0.f; // illegal box, no overlap

    // boxes in search are shifted by 1 for integral image trick. 
	// Compensate for that when intersecting with ground truth
    const Box min_intersect_box = { std::max(min_box.left,ref_box.left+1), \
                                    std::max(min_box.top,ref_box.top+1), \
                                    std::min(min_box.right,ref_box.right+1), \
                                    std::min(min_box.bottom,ref_box.bottom+1), 0. };
    if ((min_intersect_box.right< min_intersect_box.left) || (min_intersect_box.bottom < min_intersect_box.top))
        return 0.f;

    const Box max_box = { s->low[0], s->low[1], s->high[2], s->high[3], 0. };
    const double max_area = box_area(max_box);
    const double min_area = box_area(min_box);
    const double refbox_area = box_area(ref_box);

    const double min_intersection_area = box_area(min_intersect_box);
    const double ratio_denom = refbox_area + max_area - min_intersection_area;  // can be a bad estimate

    return fmin( fmax( (min_area+refbox_area)/ratio_denom-1., 0.), 1.);
}

// the loss function is Delta(y,y')= [ 1-max(box_overlap) ]_+ 
// where the max runs over all ground-truth boxes
double PyramidQualityWithLossFunction::loss_bound(const sstate* state) const {
	if (ground_truth_boxes.size() == 0) {
		return 0.;	// no ground truth means, we're in test stage, i.e. no loss term to add
	}
    // otherwise, there's a least one box

	if (ground_truth_boxes[0].score < 0) {
        return 1.; 	// negative box score -> negative example image -> use fixed loss of 1.
	}

    double max_overlap=0.;
    for (int i=0;i < ground_truth_boxes.size(); i++) {
        max_overlap = fmax( max_overlap, box_overlap_lower(state, ground_truth_boxes[i]) );
    }
    return 1-max_overlap; // simple smooth loss function. Others would be possible
}

double PyramidQualityWithLossFunction::upper_bound(const sstate* state) const {
    double quality_bound = PyramidQualityFunction::upper_bound(state);
    double loss = loss_bound(state);

    return quality_bound + loss;
}
