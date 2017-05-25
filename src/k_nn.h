/*
 *  *  * k_nn.h
 *   *   *
 *    *    *  Created on: Dec 11, 2015
 *     *     *      Author: HuanFENG
 *      *      */

#define KNN 2

#include <stdio.h>
#include <fstream>
using namespace std;

#include "bfs_index.h"          // 0
#include "pca_index.h"          // 1
#include "rkd_index.h"          // 2
#include "scf_index.h"          // 3
#include "rbc_index.h"          // 4
#include "sonnet_index.h"		// 5
#include <unistd.h>	

class K_NN {
public:
	char name[50];
	int rows, cols;
    float *data;

	static int ALG;	
	static bool SAVE_INDEX;

	~K_NN();
	void readImg();
	void buildIndex();
	void prepare(K_NN &qimg);
	void match(K_NN &qimg, int* iindex, float* idist, const char* flag, int nthreads);
	float precision(int qrows, int* iindex);
	float precisionWithDist(float *queries, int qrows, int* iindex);
private:
	void readImgInStream();
	void readGroundTruthInStream(int qrows, int knn, int* iindex);
	void writeGroundTruthInStream(int qrows, int knn, int* iindex);
	
	BFSIndex* bfs;
	PCAIndex* pca;
	RKDIndex* rkd;
	SCFIndex* scf;
	RBCIndex* rbc;
	SONNETIndex* sonnet;
};

