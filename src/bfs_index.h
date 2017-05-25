/*
 * BFSIndex.h
 *
 *  Created on: Mar 20, 2014
 *      Author: ryan
 */

#ifndef BFSINDEX_H_
#define BFSINDEX_H_

#include "bfs_linear_index.h"

class BFSIndex {
public:
public:
	BFSIndex(float*, int, int);
	virtual ~BFSIndex();
	void buildIndex();
	void knnSearch(Matrix<float>&, Matrix<int>&, Matrix<float>&, int,
			int);

private:
	LinearIndex_m<L2<float> >* index;
};

#endif /* BFSINDEX_H_ */
