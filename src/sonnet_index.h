/*
 * sonnet.cpp
 *
 *  Created on: March 24, 2017
 *      Author: Huan FENG
 */

#ifndef SONNETINDEX_H_
#define SONNETINDEX_H_

#include "common.h"

struct List
{
    int index;
    float value;
};

class SONNETIndex {
public:
	static float FRAC;
	SONNETIndex(float* data, int rows, int cols);
        virtual ~SONNETIndex();
	
	void buildIndex();
	void knnSearch(Matrix<float>& que, Matrix<int>& iindex, Matrix<float>& idist, int knn, int threads);

private:
        float *data;
        int rows, cols;
	vector<List* > Lists;
	int seekNext(float q, int &low, int &high, List* Lj);
	void sequentialSearch(float* query, int* qptr, int* iin, float* iidist, int knn);
	bool checkStatus(vector<bool> finished);
};

#endif /* SONNETINDEX_H_ */
