/*
 * RKDIndex.h
 *
 *  Created on: Mar 20, 2014
 *      Author: ryan
 */

#ifndef RKDINDEX_H_
#define RKDINDEX_H_

#include "rkd_kdtree_index.h"

class RKDIndex {
public:
	static int TREES;
	static int CHECKS;
	RKDIndex(float*data, int rows, int cols);
	virtual ~RKDIndex();
	void buildIndex();
	void knnSearch(Matrix<float>& que, Matrix<int>& iindex, Matrix<float>&idist, int knn,
			int threads);

private:
	KDTreeIndex_m<L2<float> >* index;
};

#endif /* RKDINDEX_H_ */
