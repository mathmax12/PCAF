/*
 * sonnet.cpp
 *
 *  Created on: March 24, 2017
 *      Author: Huan FENG
 */

#include "sonnet_index.h"
#include <omp.h>

float SONNETIndex::FRAC = 1.0;
struct QueueElement
{
	int qrow;
    	int index;
    	float distance;
	QueueElement(){}
	QueueElement(int _index, float _dist){
		index= _index;
		distance = _dist;
	}	
};

int compare( const void* a, const void* b )
{
	const List *l1 = (const List *)a;
	const List *l2 = (const List *)b;
	if (l1->value < l2->value) return 1;
	if (l1->value > l2->value) return -1;
	return 0;

}

SONNETIndex::SONNETIndex(float* data_, int rows_, int cols_) {
        data = data_;
        rows = rows_;
        cols = cols_;
	Lists.resize(cols, NULL);
}

SONNETIndex::~SONNETIndex(){
        delete[] data;
	Lists.clear();
}

void SONNETIndex::buildIndex(){
#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < cols; i++){
                List *Li = new List[rows];
                for(int row = 0; row < rows; row++){
                        Li[row].index = row;
                        Li[row].value = data[row * cols + i];
                }
                qsort(Li, rows, sizeof(List), compare);
		Lists[i] = Li;
        }
}

void SONNETIndex::knnSearch(Matrix<float>& que, Matrix<int>& iindex, Matrix<float>& idist, int knn, int threads){
        int qrows = que.rows;
	// iptr = [ilow, ihigh] cols * qrows;
        int* iptr = new int[qrows*cols * 2];

        // query for the Ij,h and Ij,l
 #pragma omp parallel for num_threads(threads) schedule(dynamic)
        for(int j = 0; j < cols; j++){
                List *Lj = Lists[j];
		for(int qrow = 0; qrow < qrows; qrow++){
                        float q = que[qrow][j];
                        int row = 0;
                        while(row < rows && Lj[row].value > q)
                                row++;
			//update ilow;
			iptr[(qrow*cols+j)*2] = (row==rows?-1:row);
			//update ihigh;
			iptr[(qrow*cols+j)*2+1] = row -1; 
                }
        }
// sequential searching
	if(threads == 1){
		int* qptr = iptr;
		for(int qrow = 0; qrow< qrows; qrow++, qptr+=cols*2)
			sequentialSearch(que[qrow], qptr, iindex[qrow], idist[qrow], knn);	
		return;
	}

// parallel seraching
	int* pindex;
	pindex = new int[threads];
        for(int i = 0; i < threads; i++){
                int gap = cols/threads + (cols%threads/(i+1)>0?1:0);
		pindex[i] = (i>0?pindex[i-1]:-1) + gap;
        }
//	TIMER_T runtime[2];
//	TIMER_READ(runtime[0]); 
	bool* Visited = new bool[qrows*rows];
	memset(Visited, false, qrows*rows*sizeof(bool));
	int psize = qrows * threads;
	size_t* indexes = new size_t[psize * knn];
	Matrix<size_t> inds(indexes, psize, knn);
	float* distances = new float[psize * knn];
	Matrix<float> dists(distances, psize, knn);
	int max = ceil(ceil(rows*FRAC)/threads);
#pragma omp parallel num_threads(threads)
{	
	int tid = omp_get_thread_num();
	for(int qrow = 0; qrow < qrows; qrow++){
		int counts = 0;
		int* qptr = iptr+qrow*cols*2;
		float* query = que[qrow];
		bool terminate = false;
		KNNSimpleResultSet<float> results(knn);
		while(!terminate){
			for(int j = (tid>0?pindex[tid-1]+1:0); j < pindex[tid]+1; j++){
				List *Lj = Lists[j];
				int next = seekNext(query[j], qptr[2*j], qptr[2*j+1], Lj);
				if(next == rows){
                                terminate = true;
                                break;
                       		 }
#pragma omp flush(Visited)
                        	if(!Visited[qrow*rows+next]){
                                float *feature = data;
                                feature += next*cols;
                                float dist = vdistance_(feature, que[qrow], cols);
                                Visited[qrow*rows+next] = true;
#pragma omp flush(Visited)
                                results.addPoint(dist, next);
                                if(!(counts < max)) {
                                        terminate = true;
                                        break;
                                }
                                ++counts;
                        	}//DONE check visited and calculate
                	}//DONE for loop
		}//DONE while loop
		results.copy(inds[qrow*threads+tid], dists[qrow*threads+tid], knn, false);
	    results.clear();
        }//DONE for loop
}//DONE parallel section

	for(int qrow = 0; qrow<qrows; qrow++){
		KNNSimpleResultSet<float> results(knn);
		for(int i = 0; i<threads; ++i)
			for (int k = 0; k < knn; k++)
				results.addPoint(dists[qrow*threads+i][k], inds[qrow*threads+i][k]);	
		size_t* ii = new size_t[knn];
        	results.copy(ii, idist[qrow], knn, false);
        	for (int j = 0; j < knn; ++j)
                	iindex[qrow][j] = ii[j];
        	delete[] ii;
	}
	
	delete[] distances;
	delete[] indexes;
	delete[] Visited;

//	TIMER_READ(runtime[1]);
//	printf("%f\t", TIMER_DIFF_SECONDS(runtime[0], runtime[1]));
}

int SONNETIndex::seekNext(float q, int &low, int &high, List* Lj){
	int next = 0;
        if(low < 0 && high < 0){
//		printf("= List fully checked! (If occurs in sequential case, make sure FRAC <= 1.0)\n");
                next = rows;	
        }else{
		if( high < 0){
			next = Lj[low].index;
                        low = (low<rows-1)?low+1:-1;
                 }else if(low < 0){
			next = Lj[high].index;
                        --high;
                 }else if( !(abs(Lj[low].value - q) > abs(Lj[high].value - q)) ){
                        next = Lj[low].index;
                        low = (low<rows-1)?low+1:-1;
                 }else{
                        next = Lj[high].index;
                        --high;
                 }
	}
	return next;
}

void SONNETIndex::sequentialSearch(float* query, int* qptr, int* iin, float* iidist, int knn){
	KNNSimpleResultSet<float> results(knn);
	vector<bool> Visited(rows, false);
	int count = 0;
	bool terminate = false;
	while(!terminate){
		for(int j = 0; j < cols; j++){
			List *Lj = Lists[j];
			int next = seekNext(query[j], qptr[2*j], qptr[2*j+1], Lj);
			if(Visited[next]) continue;
			
			float* feature = data;
			feature += next*cols;
			float dist = vdistance_(feature, query, cols);
			results.addPoint(dist, next);
			Visited[next] = true;
			count++;
			if(!(count < floor(rows * FRAC))) {
				terminate = true;
				break;
			}
		}
	}
	size_t* ii = new size_t[knn];
	results.copy(ii, iidist, knn, false);
	for (int j = 0; j < knn; ++j) 
		iin[j] = ii[j];
	delete[] ii;
	results.clear();
	Visited.clear();	
}
