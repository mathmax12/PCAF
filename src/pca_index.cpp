/*
 *  * PCAIndex.cpp
 *   *
 *    *  Created on: Feb 13, 2016
 *     *      Author: Huan FENG
 *      */

#include "pca_index.h"
#include "pca_svd_index.h"

int PCAIndex::TYPE = SVD_SAVED;
float PCAIndex::VALUE = 1.0;
int PCAIndex::HEAP = 1;
int PCAIndex::BLOCK = 1;
bool PCAIndex::SHOW_FILTER = 0;
SVDMatrix svdMatrix;

PCAIndex::PCAIndex(float* data_, int rows_, int cols_) {
        data = data_;
        rows = rows_;
        cols = cols_;
}

PCAIndex::~PCAIndex(){
	delete[] data;
	delete[] data_svd;
	delete[] que_data_svd;
	svdMatrix = SVDMatrix(); 
}

void PCAIndex::buildSVD() {
	SVD *svd = new SVD(data, rows, cols);
	SVDParams par;
	par.type = PCAIndex::TYPE;
	par.value = PCAIndex::VALUE;
	data_svd = svd->computeSVD(svdMatrix, par);
	svd->~SVD();
}

void PCAIndex::prepare(float* que_data, int qrows, int qcols){
	SVD *svd = new SVD(que_data, qrows, qcols);
        que_data_svd = svd->transformSVD(svdMatrix);
        svd->~SVD();
}

float* PCAIndex::prepare(float* que){
	int svd_col = svdMatrix.umatrix.rows();
	float* que_svd = new float[svd_col];
	for(int i = 0; i < svd_col; i++){
		que_svd[i] = 0.0;
		for (int j = 0; j < svdMatrix.means.cols(); j++)
			que_svd[i] += (que[j] - svdMatrix.means(j)) * svdMatrix.umatrix(i,j);
	}
	return que_svd;
}

void PCAIndex::knnSearchOnPCA(Matrix<float>& que, Matrix<int>& iindex,
				Matrix<float>& idist, int knn, int threads){
	int svd_cols = svdMatrix.umatrix.rows();
    que_data_svd = new float[que.rows * svd_cols * sizeof(float)];
    memset(que_data_svd, 0.0, que.rows*svd_cols*sizeof(float));

#pragma omp parallel num_threads(threads)
{
    #pragma omp for schedule(guided)
    for(int i = 0; i < que.rows; ++i){
        float* query_svd = que_data_svd + i * svd_cols;
        float *temp = prepare(que[i]);
        memcpy(query_svd, temp, svd_cols * sizeof(float));
        delete[] temp;
    }
	#pragma omp for schedule(dynamic)
	for(int i = 0; i < que.rows; ++i){
		float* query_svd = que_data_svd + i * svd_cols;
		float* feature_svd = data_svd;
		KNNSimpleResultSet<float> results(knn);
		for(int j = 0; j < rows; ++j, feature_svd+=svd_cols){
			float svd_dist = vdistance_(feature_svd, query_svd, svd_cols); 
			results.addPoint(svd_dist, j);
		}
		size_t* ii = new size_t[knn];
        results.copy(ii, idist[i], knn, false);
        int* iin = iindex[i];
        for (int j = 0; j < knn; ++j) {
            iin[j] = ii[j];
        }
        delete[] ii;
        results.clear();
	}
}	
	
}

void PCAIndex::knnSearch(Matrix<float>& que, Matrix<int>& iindex,
                Matrix<float>& idist, int knn, int threads){

	int svd_cols = svdMatrix.umatrix.rows();
	que_data_svd = new float[que.rows * svd_cols * sizeof(float)];
	memset(que_data_svd, 0.0, que.rows*svd_cols*sizeof(float));
	int blocks = PCAIndex::BLOCK;
	int heap = PCAIndex::HEAP;

	int psize = que.rows * blocks;
    size_t* indexes = new size_t[psize * knn];
    Matrix<size_t> inds(indexes, psize, knn);
    float* distances = new float[psize * knn];
    Matrix<float> dists(distances, psize, knn);
    int size = rows / blocks;
	
	int* filterCount;
	bool show_filter = PCAIndex::SHOW_FILTER;
	if(show_filter)
		filterCount = new int[psize];
#pragma omp parallel num_threads(threads)
{
	#pragma omp for schedule(guided)
	for(int i = 0; i < que.rows; ++i){
		float* query_svd = que_data_svd + i * svd_cols;
		float *temp = prepare(que[i]);
		memcpy(query_svd, temp, svd_cols * sizeof(float));
		delete[] temp;
	}

    #pragma omp for schedule(dynamic)
    for(int i = 0; i < psize; ++i){
	    int qrow_index = i % que.rows;
		float* query = que[qrow_index];
		float* query_svd = que_data_svd + qrow_index * svd_cols;

		int block_index = i / que.rows;
		int frow_index = block_index * size;
		
		float* feature = data;
		feature += frow_index * cols;
	    float* feature_svd = data_svd;
		feature_svd += frow_index * svd_cols;

		KNNSimpleResultSet<float> results(knn);
		float worst = results.worstDist();
	    KNNSimpleResultSet<float> results_svd(knn * heap);
		float worst_svd = results_svd.worstDist();

		float filtered = 0;
		int query_block_inner_size = block_index < (blocks-1) ? size : rows - frow_index;
	    for (int j = 0; j < query_block_inner_size; j++, feature_svd += svd_cols, feature += cols) {
		    float dist_svd = vdistance_(feature_svd, query_svd, svd_cols);
			if (dist_svd < worst_svd){
				float dist = vdistance_(feature, query, cols);
	            if (dist < worst){
		            results_svd.addPoint(dist_svd, frow_index + j);
					worst_svd = results_svd.worstDist();
				    results.addPoint(dist, frow_index + j);
					worst = results.worstDist();
	            }
		    }else
			    ++filtered;
	    }
		results.copy(inds[i], dists[i], knn, false);
	    results.clear();
		results_svd.clear();
		if(show_filter)
			filterCount[i] = filtered;
	}
        
	#pragma omp for schedule(guided)
	for(int i = 0; i < que.rows; i++){
        KNNSimpleResultSet<float> results(knn);
        for(int j = 0; j < blocks; j++)
            for (int k = 0; k < knn; k++)
                results.addPoint(dists[j * que.rows + i][k], inds[j * que.rows + i][k]);
        size_t* ii = new size_t[knn];
        results.copy(ii, idist[i], knn, false);
        int* iin = iindex[i];
        for (int j = 0; j < knn; ++j) {
            iin[j] = ii[j];
        }
        delete[] ii;
        results.clear();
    }
}
	if(show_filter){
        int filtered = 0;
    	for(int i = 0; i < psize; i++)
			filtered += filterCount[i];
        printf("Filtering rate: %f \%\n", filtered*100.0/que.rows/rows);
		delete[] filterCount;
    }

    delete[] indexes;
    delete[] distances;
}

