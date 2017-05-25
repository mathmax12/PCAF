/*
 * rbc_index.cpp
 *
 *  Created on: Oct 9, 2013
 *      Author: ryan
 */

#include "rbc_index.h"
#include "omp.h"

int RBCIndex::REP = 1;

RBCIndex::RBCIndex(float* data_, int rows_, int cols_) {
	ddata = data_;
	drows = rows_;
	dcols = cols_;
	ri = NULL;
	shuf = NULL;
}

RBCIndex::~RBCIndex() {
	ddata = NULL;
	drows = dcols = 0;
}

void RBCIndex::buildIndex() {
	initMat(&x, drows, dcols);
	x.mat = ddata;
	int reps = sqrt(drows) * REP;
	printf("Number of representatives:%d\n", reps);
//	reps = 512;

	ri = (rep*) calloc(CPAD(reps), sizeof(*ri));

//	buildExact(x, &r, ri, reps);
	shuf = buildOneShot(x, &r, ri, reps);
}

void RBCIndex::knnSearch(Matrix<float>& que, Matrix<int>& iindex,
		Matrix<float>& idist, int knn, int threads) {
	unint rows = que.rows, cols = que.cols;
	omp_set_num_threads(threads);
	matrix q;
	initMat(&q, rows, cols);
	q.mat = que.ptr();
	unint **NNs = (unint**) calloc(rows, sizeof(*NNs));
	real **dToNNs = (real**) calloc(rows, sizeof(*dToNNs));

	for (unint i = 0; i < rows; i++) {
		NNs[i] = (unint*) iindex[i];
		dToNNs[i] = (real*) idist[i];
	}

	searchOneShotK(q, x, r, ri, NNs, dToNNs, knn);
//	searchOneShotK(q, x, r, ri, NNs, dToNNs, knn);
//	searchExactK(q, x, r, ri, NNs, dToNNs, knn);
//	searchMultiShootK(q, x, r, ri, NNs, dToNNs, shoot, knn);

	free(NNs);
	free(dToNNs);
}

void RBCIndex::searchOneShotK(matrix q, matrix x, matrix r, rep *ri,
		unint **NNs, real **dNNs, unint K) {
	unint *repID = (unint*) calloc(q.pr, sizeof(*repID));
	real *dT = (real*) calloc(q.pr, sizeof(*dT));

	// Determine which rep each query is closest to.
	brutePar(r, q, repID, dT);

	// Search that rep's ownership list.
	bruteMapK(x, q, ri, repID, NNs, dNNs, K);

	free(repID);
	free(dT);

}

void RBCIndex::brutePar(matrix X, matrix Q, unint *NNs, real *dToNNs) {
	uint i, j, k, t;

#pragma omp parallel for private(t,k,j) schedule(dynamic)
	for (i = 0; i < Q.pr / CL; i++) {
		t = i * CL;
		for (j = 0; j < CL && t + j < Q.r; j++) {
			dToNNs[t + j] = MAX_REAL;
			NNs[t + j] = 0;
		}

		float dist = 0;
		for (j = 0; j < X.r; j++) {
			for (k = 0; k < CL; k++) {
				if (t + k < Q.r) {
//					dist = distVec(Q, X, t + k, j);
					dist = vdistance_(&Q.mat[(t + k) * Q.ld], &X.mat[j * X.ld],
							Q.pc);
					if (dist < dToNNs[t + k]) {
						NNs[t + k] = j;
						dToNNs[t + k] = dist;
					}
				}
			}
		}
	}
}

void RBCIndex::bruteMapK(matrix X, matrix Q, rep *ri, unint* qMap,
		unint **NNs, real **dToNNs, unint K) {
	uint i, j, k;

	size_t *qSort = (size_t*) calloc(Q.pr, sizeof(*qSort));
	gsl_sort_uint_index(qSort, qMap, 1, Q.r);

	unint nt = omp_get_max_threads();
	heap **hp;
	hp = (heap**) calloc(nt, sizeof(*hp));
	for (i = 0; i < nt; i++) {
		hp[i] = (heap*) calloc(CL, sizeof(**hp));
		for (j = 0; j < CL; j++)
			createHeap(&hp[i][j], K);
	}

#pragma omp parallel for private(j,k) schedule(dynamic)
	for (i = 0; i < Q.pr / CL; i++) {
		unint row = i * CL;
		unint tn = omp_get_thread_num();
		heapEl newEl;

		rep rt[CL];
		unint maxLen = 0;
		for (j = 0; j < CL; j++) {
			rt[j] = ri[qMap[qSort[row + j]]];
			maxLen = MAX(maxLen, rt[j].len);
		}

		real dist = 0;
		for (j = 0; j < maxLen; j++) {
			for (k = 0; k < CL; k++) {
				if (j < rt[k].len) {
//					dist = distVec(Q, X, qSort[row + k], rt[k].lr[j]);
					dist = vdistance_(&Q.mat[qSort[row + k] * Q.ld],
							&X.mat[rt[k].lr[j] * X.ld], Q.pc);
					if (dist < hp[tn][k].h[0].val) {
						newEl.id = rt[k].lr[j];
						newEl.val = dist;
						replaceMax(&hp[tn][k], newEl);
					}
				}
			}
		}

		for (j = 0; j < CL; j++)
			heapSort(&hp[tn][j], NNs[qSort[row + j]], dToNNs[qSort[row + j]]);

		for (j = 0; j < CL; j++)
			reInitHeap(&hp[tn][j]);
	}

	for (i = 0; i < nt; i++) {
		for (j = 0; j < CL; j++)
			destroyHeap(&hp[i][j]);
		free(hp[i]);
	}
	free(hp);
	free(qSort);
}
