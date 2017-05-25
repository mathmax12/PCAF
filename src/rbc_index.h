/*
 * rbc_index.h
 *
 *  Created on: Oct 9, 2013
 *      Author: ryan
 */

#ifndef RBC_INDEX_H_
#define RBC_INDEX_H_

#include "rbc/utils.h"
#include "rbc/rbc.h"
#include "rbc/dists.h"
#include "flann/util/matrix.h"
#include "flann/util/params.h"
#include <gsl/gsl_sort.h>
using namespace flann;

#include "common.h"

class RBCIndex {
public:
	static int REP;
	RBCIndex(float*, int, int);
	virtual ~RBCIndex();
	void buildIndex();
	void knnSearch(Matrix<float>&, Matrix<int>&, Matrix<float>&, int,
			int);

private:
	void searchOneShotK(matrix, matrix, matrix, rep*, unint**, real**, unint);
	void brutePar(matrix, matrix, unint*, real*);
	void bruteMapK(matrix, matrix, rep*, unint*, unint**, real**, unint);
	float* ddata;
	int drows, dcols;
	matrix x, r;
	rep *ri;
	unint* shuf;
};

#endif /* RBC_INDEX_H_ */
