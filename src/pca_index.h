/*
*  * PCAIndex.h
*   *
*    *  Created on: Feb 13, 2016
*     *      Author: Huan FENG
*      */

#ifndef PCAINDEX_H_
#define PCAINDEX_H_

#include "common.h"
using namespace flann;

class PCAIndex {
public:

	static int TYPE, HEAP, BLOCK;
	static float VALUE;
	static bool SHOW_FILTER;
 
    PCAIndex(float* data, int rows, int cols);
    virtual ~PCAIndex();
    void buildSVD();
	void prepare(float* que_data, int qrows, int qcols);
    void knnSearch(Matrix<float>& que, Matrix<int>& iindex, Matrix<float>& idist, int knn,
        	int threads);
	void PCAIndex::knnSearchOnPCA(Matrix<float>& que, Matrix<int>& iindex,
                Matrix<float>& idist, int knn, int threads);
private:
	float *data, *data_svd, *que_data_svd;
	int rows, cols;
	float* prepare(float* que);
};

#endif /* PCAINDEX_H_ */
