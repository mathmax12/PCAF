#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "common.h"

#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/Core>
using Eigen::MatrixXf;
using Eigen::RowVectorXf;

#define SVD_SAVED		0
#define SVD_APPOINT_DIMS	1
#define SVD_APPOINT_PERCENTAGE	2
#define SVD_RETAIN_VARIATION	3

struct  SVDParams {
	int type;
	float value;
	SVDParams(){
		type = SVD_RETAIN_VARIATION;
		value = 1.0;
	}
};

struct SVDMatrix {
	RowVectorXf means;
	MatrixXf umatrix;
	SVDMatrix() {
		means.resize(1, 0);
		umatrix.resize(0, 0);
	}
};

class SVD {
public:
	SVD(float* data, int rows, int cols){
		mat = Eigen::Map<MatrixXf>(data, cols, rows).transpose();
	}
	
	~SVD() {
		mat.resize(0, 0);
	}
	
	float* computeSVD(SVDMatrix &svdMatrix, SVDParams param = SVDParams()){
		if (param.type == SVD_SAVED){
			readSVDMatrix(svdMatrix);
		}else{
			svdMatrix.means = mat.colwise().mean();
            MatrixXf centered = mat.rowwise() - svdMatrix.means;
        	MatrixXf cov = (centered.adjoint() * centered) / float(mat.rows() - 1);
			Eigen::JacobiSVD<MatrixXf> svd(cov, Eigen::ComputeFullU);
            MatrixXf S = svd.singularValues();	
			S = svd.singularValues();
			int rank = chooseRank(S, param);
			MatrixXf U = svd.matrixU();
	        svdMatrix.umatrix = reduceUTranspose(U, rank);
			saveSVDMatrix(svdMatrix);
		}
		float* svd_data = transformSVD(svdMatrix);
		
		// print messages
		printf("Dimensionality in PCA space: %d\n", svdMatrix.umatrix.rows());
//		calVariance(S, svdMatrix.umatrix.rows());
		return svd_data;
	}

	float* transformSVD(SVDMatrix svdMatrix){
        mat = mat.rowwise() - svdMatrix.means;
		MatrixXf svd_mat = (svdMatrix.umatrix * mat.transpose()).transpose();
    	float* svd_data = new float[svd_mat.rows() * svd_mat.cols()];
    	Eigen::Map<MatrixXf>(svd_data, svd_mat.cols(), svd_mat.rows()) = svd_mat.transpose(); 
		return svd_data;
	}
	
private:
	MatrixXf mat;
	void calVariance(MatrixXf S, int rank){
		float sum = S.sum();
        float tsum = 0.0;
		for(int i=0;i<rank;i++) tsum+=S(i);
        printf("Variance Retained: %f\n", tsum/sum);
	}

	int chooseRank(MatrixXf S, SVDParams param){
		int rank = 0;
		if(param.type == SVD_APPOINT_DIMS)
			rank = (int) param.value;
		else if(param.type == SVD_APPOINT_PERCENTAGE)
			rank = S.rows() * param.value;
		else {
			float sum = S.sum();
			float tsum = 0.0, variance_retained = 0.0;
			for (int i = 0; i < S.rows(); i ++){
				tsum += S(i);
				variance_retained = tsum/sum;
				if (variance_retained > param.value){
                    rank = i + 1;
            		break;
                }
			}
		}
		return rank;
	}
	
	MatrixXf reduceUTranspose(MatrixXf U, int rank){
        MatrixXf TUReduced = U.transpose().topRows(rank);
    	return TUReduced;
	}
	
	void saveSVDMatrix(SVDMatrix svdMatrix){
		ofstream out;
		out.open("index/SVDMatrix.bin");

		int rows = svdMatrix.means.rows();
		int cols = svdMatrix.means.cols();
		out.write((char*) &rows, sizeof(int));
        out.write((char*) &cols, sizeof(int));
		int size = rows * cols;
		float *mean = new float[size];
		Eigen::Map<MatrixXf>(mean, cols, rows) = svdMatrix.means.transpose();
		out.write((char*) mean, size * sizeof(float));

		rows = svdMatrix.umatrix.rows();
        cols = svdMatrix.umatrix.cols();
		size = rows * cols;
        out.write((char*) &rows, sizeof(int));
        out.write((char*) &cols, sizeof(int));
        float *u = new float[size];
        Eigen::Map<MatrixXf>(u, cols, rows) = svdMatrix.umatrix.transpose();
        out.write((char*) u, size * sizeof(float));
		out.close();
	}
	
	void readSVDMatrix(SVDMatrix &svdMatrix){
		ifstream in;
        in.open("index/SVDMatrix.bin");
	    int rows = 0;
    	int cols = 0;
	    in.read((char*) &rows, sizeof(int));
    	in.read((char*) &cols, sizeof(int));
        int size = rows * cols;
	    float* mean = new float[size];
	    in.read((char*) mean, size * sizeof(float));
		svdMatrix.means = Eigen::Map<MatrixXf>(mean, cols, rows).transpose();
	        
		in.read((char*) &rows, sizeof(int));
        in.read((char*) &cols, sizeof(int));
		size = rows * cols;
        float* mat_data = new float[size];
        in.read((char*) mat_data, size * sizeof(float));
        svdMatrix.umatrix = Eigen::Map<MatrixXf>(mat_data, cols, rows).transpose();
		in.close();
	}

};
