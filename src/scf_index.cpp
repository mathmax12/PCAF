/*
 * ckdtree.cpp
 *
 *  Created on: Feb 14, 2016
 *      Author: Huan FENG
 */

#include "scf_index.h"
#include "cmath"

#define ITER	10
#define TRAINT	10

float SCFIndex::SCALE_U = 0.0;
float SCFIndex::SCALE_D = 0.0;
int SCFIndex::SUBSPACES = 16;
int SCFIndex::CLUSTERS = 8;
bool SCFIndex::SHOW_FILTER = false;

L2<float> SCFIndex::distance_;

float SCFIndex::estimate(unsigned char* blt, float* mdists) {
	int csize = clusters.size();
	float dist = 0;
	int j = 0;
	//#pragma novector
	while (j < csize) {
		dist += mdists[blt[j++]];
		mdists += branch;
		dist += mdists[blt[j++]];
		mdists += branch;
	}
	return dist;
}

float SCFIndex::estimate(int index, float* mdists) {
	unsigned char* blt = beLongTo + index * clusters.size();
	return estimate(blt, mdists);
}

struct Compare {
	Compare(float* d) {
		distances = d;
	}
	float* distances;
	bool operator()(int i, int j) {
		return (distances[i] < distances[j]);
	}
};

bool compare(Cluster* c1, Cluster* c2) {
	return (c1->variance > c2->variance);
}

SCFIndex::SCFIndex(float* data_, int rows_, int cols_) {
	ddata = data_;
	drows = rows_;
	dcols = cols_;
	beLongTo = NULL;
	branch = 0;
}

SCFIndex::~SCFIndex() {
	ddata = NULL;
	drows = dcols = 0;
	for (size_t i = 0; i < clusters.size(); ++i) {
		clusters[i]->~Cluster();
	}
	clusters.clear();
	if (beLongTo != NULL) {
		delete[] beLongTo;
	}
}

void SCFIndex::buildIndex() {
	//build clusters:
	buildIndex(SUBSPACES, CLUSTERS, ITER);
}

void SCFIndex::buildIndex(int depth, int branch_, int iter_) {
	for (size_t i = 0; i < clusters.size(); ++i) {
		clusters[i]->~Cluster();
	}
	clusters.clear();
	if (beLongTo != NULL) {
		delete[] beLongTo;
	}
	branch = branch_;
	KMeansParams par;
	par["branching"] = branch_;
	par["iterations"] = iter_;
	int dimen = dcols / depth;
	for (int i = 0; i < depth; ++i) {
		Matrix<float> mat(ddata + dimen * i, drows, dimen,
				dcols * sizeof(float));
		KMeans<L2<float> > kmeans(mat, par);
		kmeans.buildIndex();
		Cluster* clu = kmeans.getCluster();
		clu->start = dimen * i;
		clusters.push_back(clu);

//		float ra = 0;
//		for(int j = 0; j < branch; ++j){
//			ra += clu->radiuses[j] * clu->radiuses[j];
//		}
//		ra /= branch;
//		printf("%.2f\n", ra);
	}
	std::sort(clusters.begin(), clusters.end(), compare);

	int csize = clusters.size();
	beLongTo = new unsigned char[drows * csize];
	for (int i = 0; i < csize; ++i) {
		unsigned char* blt = beLongTo + i;
		int* bel = clusters[i]->beLongTo;
		for (int j = 0; j < drows; ++j) {
			*blt = bel[j];
			blt += csize;
		}
	}
}

/*void SCFIndex::update(float pre, float precision) {
	float r1 = std::abs(scale_u - SCALE_U);
	float r2 = std::abs(scale_d - SCALE_D);
	float acc = std::abs((precision - pre));
	acc = std::min(acc, 0.15f);
	acc = std::max(acc, 0.05f);
	if (pre > precision) {
		if (r1 > r2) {
			scale_d -= acc;
		} else {
			scale_u += acc;
		}
	} else {
		if (r1 >= r2) {
			scale_d += acc;
		} else {
			scale_u -= acc;
		}
	}
//	printf("U: %.4f, D: %.4f ", scale_u, scale_d);
}
*/
void SCFIndex::knnSearch(Matrix<float>& que, Matrix<int>& iindex,
		Matrix<float>& idist, int knn, int threads) {
	int ssize = que.rows;
	int* filterCount;
        bool show_filter = SHOW_FILTER;
        if(show_filter)
                filterCount = new int[ssize];

#pragma omp parallel num_threads(threads)
{
#pragma omp for schedule(dynamic)

	for (int i = 0; i < ssize; ++i) {
		float* query = que[i];

		//Calculate distances to each clusters:
		//Calculate the minimum squared distance:
		float* min_distances = init(query);

		//Search the CKD-tree:
		KNNSimpleResultSet<float> results(knn);
		if (show_filter)
			filterCount[i] = explore(query, &results, min_distances);
		else
			explore(query, &results, min_distances);

		size_t* ii = new size_t[knn];
		results.copy(ii, idist[i], knn, false);
		int* iin = iindex[i];
		for (int j = 0; j < knn; ++j) {
			iin[j] = ii[j];
		}

		delete[] ii;
		delete[] min_distances;
	}
}	
	if (show_filter){
                int filtered = 0;
                for(int i = 0; i < ssize; i++)
                        filtered += filterCount[i];
                printf("Filtering rate: %f \%\n", filtered*100.0/que.rows/drows);
                delete[] filterCount;
        }
}

int SCFIndex::explore(float* query, ResultSet<float>* results, float* mdists) {
	int csize = clusters.size();
	unsigned char* blt = beLongTo;
	float worst = results->worstDist();
	float* feature = ddata;
	
	int filter = 0;
	for (int i = 0; i < drows; ++i, blt += csize, feature += dcols) {
		float dist = estimate(blt, mdists);
		if (dist >= worst) {
			++filter;
			continue;
		}
		dist = vdistance_(feature, query, dcols);
		results->addPoint(dist, i);
		worst = results->worstDist();
	}
	return filter;
}

/*int SCFIndex::getClusterSize() {
	return clusters.size() * branch;
}

int SCFIndex::getClusterNum(){
	return clusters.size();
}*/

/*unsigned char* SCFIndex::getBlt(float* center, unsigned char* blt) {
	int csize = clusters.size();
	if(blt == NULL){
		blt = new unsigned char[csize];
	}
	for (int j = 0; j < csize; ++j) {
		Cluster* clu = clusters[j];
		float* que = center + clu->start;
		unsigned char index = 0;
		float max = std::numeric_limits<float>::max();
		for (int m = 0; m < branch; ++m) {
			float* cen = clu->centers + m * clu->cols;
			float dist = vdistance_(que, cen, clu->cols);
			if(dist < max){
				index = m;
				max = dist;
			}
		}
		blt[j] = index;
	}
	return blt;
}
*/
float* SCFIndex::init(float* query, float* min_distances) {
	int csize = clusters.size();
	if (min_distances == NULL) {
		min_distances = new float[csize * branch];
	}

	float* min_dists = min_distances;
	for (int j = 0; j < csize; ++j, min_dists += branch) {
		Cluster* clu = clusters[j];
		float* que = query + clu->start;
		float* cen = clu->centers;
		for (int m = 0; m < branch; ++m, cen += clu->cols) {
			float d = vdistance_(que, cen, clu->cols);

			float v = d / clu->radiuses[m] / clu->radiuses[m] - 1;
			float tmp = 1;
			if (v <= 0) {
				tmp += SCALE_D * v;
			} else {
				tmp += SCALE_U * v;
			}
			d *= tmp * tmp;

			min_dists[m] = d;
		}
	}
	return min_distances;
}
/*
void SCFIndex::getStatics(int s, int c, int iter, float* results,
		float* queries, int qrows, int knn) {
	TIMER_T start, end;
	TIMER_READ(start);
	buildIndex(s, c, iter);
	TIMER_READ(end);
//	printf("%.1f & ", TIMER_DIFF_SECONDS(start, end));

	int csize = clusters.size();
	float* distances = new float[csize * branch];
	float* estimated = new float[drows];
	float small_average = 0;
	float average_average = 0;
	float standard_average = 0;

	float est_time = 0;
	KNNSimpleResultSet<float> heap(knn);
	size_t* index1 = new size_t[knn];
	float* dist1 = new float[knn];
	size_t* index2 = new size_t[knn];
	float* dist2 = new float[knn];
	int filtered = 0;
	int correct = 0;

	for (int q = 0; q < qrows; ++q) {
		float* query = queries + q * dcols;

		TIMER_READ(start);
		for (int i = 0; i < csize; ++i) {
			Cluster* clu = clusters[i];
			float* que = query + clu->start;
			float* dists = distances + i * branch;
			for (int j = 0; j < branch; ++j) {
				float* cen = clu->centers + j * clu->cols;
				float d = vdistance_(que, cen, clu->cols);

				float v = d / clu->radiuses[j] / clu->radiuses[j];
				float tmp = 0;
				if (v <= 1) {
					tmp = 1 + scale_d * (v - 1);
				} else {
					tmp = 1 + scale_u * (v - 1);
				}
				d *= tmp * tmp;

				dists[j] = d;
			}
		}

		unsigned char* blt = beLongTo;
		for (int i = 0; i < drows; ++i, blt += csize) {
			estimated[i] = estimate(blt, distances);
		}
		TIMER_READ(end);
		est_time += TIMER_DIFF_SECONDS(start, end);
		for (int i = 0; i < drows; ++i) {
			estimated[i] = sqrt(estimated[i]);
		}

		float* result = results + q * drows;

		int small = 0;
		for (int i = 0; i < drows; ++i) {
			if (estimated[i] <= result[i]) {
				++small;
			}
		}
		small_average += small * 1.0f / drows;

		float average = 0;
		for (int i = 0; i < drows; ++i) {
			average += abs(result[i] - estimated[i]);
		}
		average /= drows;
		average_average += average;

		float standard = 0;
		for (int i = 0; i < drows; ++i) {
			float tmp = abs(result[i] - estimated[i]);
			tmp -= average;
			standard += tmp * tmp;
		}
		standard /= drows;
		standard = sqrt(standard);
		standard_average += standard;

		heap.clear();
		for (int i = 0; i < drows; ++i) {
			heap.addPoint(result[i], i);
		}
		heap.copy(index1, dist1, knn, false);

		heap.clear();
		for (int i = 0; i < drows; ++i) {
			if (estimated[i] <= heap.worstDist()) {
				heap.addPoint(result[i], i);
			} else {
				++filtered;
			}
		}
		heap.copy(index2, dist2, knn, false);

		int cur1 = 0, cur2 = 0;
		while (cur1 < knn) {
			if (index1[cur1] != index2[cur2]) {
				++cur1;
			} else {
				++cur1;
				++cur2;
				++correct;
			}
		}
	}
	small_average /= qrows;
	average_average /= qrows;
	standard_average /= qrows;
//	printf("%.4f %.2f %.2f\n", small_average, average_average,
//			standard_average);

//	printf("%.1f & ", est_time);
//	printf("%.1f & ", average_average);
//	printf("%.1f\\%% & ", filtered * 100.0f / drows / qrows);
//	printf("%.1f\\%%\n", correct * 100.0f / knn / qrows);
	printf("%.1f %.1f ", scale_u, scale_d);
	printf("%.4f ", filtered * 1.0f / drows / qrows);
	printf("%.4f\n", correct * 1.0f / knn / qrows);

	delete[] distances;
	delete[] estimated;
	delete[] index1;
	delete[] dist1;
	delete[] index2;
	delete[] dist2;
}*/
