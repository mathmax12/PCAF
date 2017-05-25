/*
 * RKDIndex.cpp
 *
 *  Created on: Mar 20, 2014
 *      Author: ryan
 */

#include "rkd_index.h"

int RKDIndex::TREES = 0;
int RKDIndex::CHECKS = 64;

RKDIndex::RKDIndex(float* data_, int rows_, int cols_){
	Matrix<float> mat(data_, rows_, cols_);
	index = new KDTreeIndex_m<L2<float> >(mat, KDTreeIndex_mParams(TREES));
}

RKDIndex::~RKDIndex(){
	delete index;
}

void RKDIndex::buildIndex(){
	const char *name = "index/RKDIndex.bin";
	if (TREES > 0){
		index->buildIndex();
/*		FILE *in;
		in = fopen(name, "w");
		index->saveIndex(in);
		fclose(in);
	}else{
		FILE *out;
		out = fopen(name, "r");
		index->loadIndex(out);
		fclose(out);
//		printf(index.trees);
*/	}
}

void RKDIndex::knnSearch(Matrix<float>& que, Matrix<int>& iindex,
		Matrix<float>& idist, int knn, int threads) {
	SearchParams par;
	par.cores = threads;
	par.checks = CHECKS;
	index->knnSearch(que, iindex, idist, knn, par);
}


