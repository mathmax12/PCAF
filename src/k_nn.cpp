/*
 *  *  *  *  * k_nn.cpp
 *   *   *   *   *
 *    *    *    *    *  Created on: Dec 11, 2015
 *     *     *     *     *      Author: HuanFENG
 *      *      *      *      */

#include "k_nn.h"
#include "common.h"
int K_NN::ALG = 0;

K_NN::~K_NN() {
	delete[] data;
    switch (ALG) {
		case 0:
			delete bfs;
			break;
		case 1:
			delete pca;
			break;
		case 2:
			delete rkd;
			break;
		case 3:
			delete scf;
			break;
		case 4:
			delete rbc;
			break;
		case 5:
			delete sonnet;
			break;
		default:
			break;
	}
	rows = 0;
	cols = 0;
}

void K_NN::buildIndex(){
	switch (ALG) {
        case 0:
		bfs = new BFSIndex(data, rows, cols);                	
		bfs->buildIndex();
                break;
	case 1:
		pca = new PCAIndex(data, rows, cols);
        pca->buildSVD();
		break;
	case 2:
        rkd = new RKDIndex(data, rows, cols);
        rkd->buildIndex();
		break;
	case 3:
       	scf = new SCFIndex(data, rows, cols);
        scf->buildIndex();
		break;
	case 4:
		rbc = new RBCIndex(data, rows, cols);
		rbc->buildIndex();
		break;
	case 5:
		sonnet = new SONNETIndex(data, rows, cols);
		sonnet->buildIndex();
		break;
	default:
		printf("=== WARNING:(KNN) No algorithm chosen");
		break;		
	}
}

void K_NN::prepare(K_NN &qimg){
	switch(ALG){
	case 1:
		pca->prepare(qimg.data, qimg.rows, qimg.cols);
		break;
	default:
		break;
	}
}

void K_NN::match(K_NN &qimg, int* iindex, float* idist, const char* flag, int nthreads) {
	int knn = KNN;
    float *query = qimg.data;
    Matrix<float> que(query, qimg.rows, qimg.cols);
    Matrix<int> ind(iindex, qimg.rows, knn);
	Matrix<float> dis(idist, qimg.rows, knn);
	switch (ALG) {
	case 0:
		bfs->knnSearch(que, ind, dis, knn, nthreads);
		break;
	case 1:
		if(strcmp(flag, "f") == 0)
			PCAIndex::SHOW_FILTER = true;
		else
			PCAIndex::SHOW_FILTER = false;

		pca->knnSearch(que, ind, dis, knn, nthreads);
		break;
	case 2:
		rkd->knnSearch(que, ind, dis, knn, nthreads);
		break;
	case 3:
		if(strcmp(flag, "f") == 0)
            SCFIndex::SHOW_FILTER = true;
        else
            SCFIndex::SHOW_FILTER = false;
		scf->knnSearch(que, ind, dis, knn, nthreads);
		break;
	case 4:
		rbc->knnSearch(que, ind, dis, knn, nthreads);
		break;
	case 5:
		sonnet->knnSearch(que, ind, dis, knn, nthreads);
		break;
	default:
		break;
	}	
	if(strcmp(flag, "w") == 0)
                writeGroundTruthInStream(qimg.rows, knn, iindex);
}

float K_NN::precisionWithDist(float* queries, int qrows, int* iindex){
    int size = qrows * KNN;
    int* index = new int[size];
    readGroundTruthInStream(qrows, KNN, index);

/*	int* sindex = new int[size];
	char* filename = "../features/sonnet_results.txt";
	ifstream infile(filename, ios::in);
	if (!infile) {
		cout << filename << " could not open" << endl;
		exit(1);
	}
	int i = 0;
	int d = 0;
	while (!infile.eof()) {
		infile >> d;
		sindex[i] = d;
		if(i ==0) cout << sindex[i] << endl;
		i++;
	}
*///	Matrix<int> ind(sindex, qrows, KNN);
    Matrix<int> eind(index, qrows, KNN);
	Matrix<int> ind(iindex, qrows, KNN);
	int num = 0;
    for (int i = 0; i < qrows; ++i) {
        int* indx1 = eind[i];
        int* indx2 = ind[i];
        int cur1 = 0, cur2 = 0;
        while (cur1 < KNN) {
            cur2 = 0;
            while(cur2 < KNN){
				if (indx1[cur1] == indx2[cur2])
                    ++num;
				else{
					float* q = queries+i*cols;
					float* dcur1 = data+indx1[cur1]*cols;
					float* dcur2 = data+indx2[cur2]*cols;
					if(vdistance_(q, dcur1, cols) == vdistance_(q, dcur2, cols)){
						++num;
						++cur1;}
				}
				++cur2;
			}
			++cur1;
		}
	}
    // pre - precision
	//printf("num %d qrows %d size %d\n", num, qrows, size);
	float pre = num * 1.0f / size * 100.00;
	delete[] index;
	return(pre);
}


float K_NN::precision(int qrows, int* iindex){
	int size = qrows * KNN;
    int* index = new int[size];
    readGroundTruthInStream(qrows, KNN, index);
	Matrix<int> eind(index, qrows, KNN);
	Matrix<int> ind(iindex, qrows, KNN);
	int num = 0;
    for (int i = 0; i < qrows; ++i) {
		int* indx1 = eind[i];
        int* indx2 = ind[i];
        int cur1 = 0, cur2 = 0;
        while (cur1 < KNN) {
			cur2 = 0;
			while(cur2 < KNN){
				if (indx1[cur1] == indx2[cur2])
					++num;
				++cur2;
			}
			++cur1;
		}
	}
	// pre - precision
	//printf("num %d qrows %d size %d\n", num, qrows, size);
    float pre = num * 1.0f / size * 100.00;
	delete[] index;
	return(pre);
}

void K_NN::readImg(){
	readImgInStream();
}

void K_NN::readImgInStream(){
    ifstream in;
    char tname[50];
    strcpy(tname, name);
    strcat(tname, ".bin");
    in.open(tname);
    in.read((char*) &rows, sizeof(int));
    in.read((char*) &cols, sizeof(int));
    int size = rows * cols;
    data = new float[size];
    in.read((char*) data, size * sizeof(float));
    in.close();

	if (ALG == 4 && cols % 4 != 0) {
        int nrows = rows / 4 * 4;
        int ncols = cols / 4 * 4 + 4;
        size = nrows * ncols;
        float* ndata = new float[size];
        memset(ndata, 0, size * sizeof(float));
        int step1 = 0;
        int step2 = 0;
        for (int i = 0; i < nrows; ++i) {
            memcpy(ndata + step1, data + step2, cols * sizeof(float));
            step1 += ncols;
            step2 += cols;
        }
        delete[] data;
        data = ndata;
        rows = nrows;
        cols = ncols;
    }

}

void K_NN::readGroundTruthInStream(int qrows, int knn, int* iindex) {
    ifstream in;
    char tname[50];
    strcpy(tname, name);
    strcat(tname, ".exact");
    in.open(tname);

    int rrows, ccols;
    in.read((char*) &rrows, sizeof(int));
    in.read((char*) &ccols, sizeof(int));
	if (rrows != qrows || ccols != knn) {
		printf("\n!ERROR(k_nn.cpp): Not the right image pair!!\n");
		exit(0);
	}
	if (iindex == NULL) {
        printf("\n!ERROR(k_nn.cpp): Empty buffer!!\n");
        exit(0);
    }
    int size = qrows * knn;
    in.read((char*) iindex, size * sizeof(int));
    in.close();	
}

void K_NN::writeGroundTruthInStream(int qrows, int knn, int* iindex) {
    ofstream out;
    char tname[50];
    strcpy(tname, name);
    strcat(tname, ".exact");
    out.open(tname);
    out.write((char*) &qrows, sizeof(int));
    out.write((char*) &knn, sizeof(int));
    int size = qrows * knn;
    out.write((char*) iindex, size * sizeof(int));
    out.close();
}
