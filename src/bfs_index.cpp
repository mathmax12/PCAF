/*
 * BFSIndex.cpp
 *
 *  Created on: Mar 20, 2014
 *      Author: ryan
 */

#include "bfs_index.h"

BFSIndex::BFSIndex(float* data_, int rows_, int cols_){
	Matrix<float> mat(data_, rows_, cols_);
	index = new LinearIndex_m<L2<float> >(mat);
}

BFSIndex::~BFSIndex(){
	delete index;
}

void BFSIndex::buildIndex(){
	index->buildIndex();
}

void BFSIndex::knnSearch(Matrix<float>& que, Matrix<int>& iindex,
		Matrix<float>& idist, int knn, int threads) {
	SearchParams par;
	par.cores = threads;
	index->knnSearch(que, iindex, idist, knn, par);
}
