/*
 * ckdtree.h
 *
 *  Created on: Jul 22, 2013
 *      Author: xiaoxintang
 */

#ifndef SCF_KMEANS_H_
#define SCF_KMEANS_H_

#include "scf_kmeans_index.h"
#include <vector>

using namespace std;

class SCFIndex {
public:
	SCFIndex(float*, int, int);
	virtual ~SCFIndex();
	void buildIndex();
	void knnSearch(Matrix<float>&, Matrix<int>&, Matrix<float>&, int,
			int threads);
	float estimate(int, float*);
	float estimate(unsigned char*, float*);
	int explore(float*, ResultSet<float>*, float*);

	int drows, dcols, branch;
	vector<Cluster*> clusters;

	static float SCALE_U, SCALE_D;
	static int SUBSPACES, CLUSTERS;
	static bool SHOW_FILTER;

private:
	float* ddata;
	unsigned char* beLongTo;

	static L2<float> distance_;
	void buildIndex(int, int, int);
	float* init(float*, float* md = NULL);
};

#endif /* CKDTREE_H_ */
