#include <stdio.h>
#include "csv_parser.h"
#include "k_nn.h"

#define SKIP			4
#define MAX_THREADS     16
//#define EXACT
//Enable EXACT to save exact kNN for a dataset to a binary file named "dataset.exact"

int main(int argc, char** argv) {
	printf("==============================\n");
	if(argc < 4) printf("!ERROR:Inappropriate Input.\n");
	char *format = argv[1];             //features/_DSC0%s.jpg
    char *csv_file = argv[2];           //feature/sift-small.csv
    char* algs = argv[3];
    int alg = 0;
    if(strcmp(algs, "bfs") == 0){
		alg = 0;
		printf("ALGORITHM CHOSEN: [BF] FLANN Brute Rorce\n");
    }else if(strcmp(algs, "pcaf") == 0){
        alg = 1;
		printf("ALGORITHM CHOSEN: [PCAF] Principal Component Analysis based Filtering\n");
		if(argc != 8){
			printf("!ERROR(main.cpp):Inappropriate parameters.\n");
			exit(0);
		}
    }else if(strcmp(algs, "rkd") == 0){
        alg = 2;
		printf("ALGORITHM CHOSEN: [RKD] Randomized KD tree\n");
		if(argc != 6){
            printf("!ERROR(main.cpp):Inappropriate parameters.\n");
			exit(0);
        }
    }else if(strcmp(algs, "scf") == 0){
        alg = 3;
		printf("ALGORITHM CHOSEN: [SCF] Subspace Clusters based Filtering\n");
		if(argc != 8){
            printf("!ERROR(main.cpp):Inappropriate parameters.\n");
			exit(0);
		}
    }else if(strcmp(algs, "rbc") == 0){
        alg = 4;
		printf("ALGORITHM CHOSEN: [RBC] Random Ball Cover\n");
		if(argc != 5){
            printf("!ERROR(main.cpp):Inappropriate parameters.\n");
        }
    }else if(strcmp(algs, "sonnet") == 0){
        alg = 5;
		printf("ALGORITHM CHOSEN: [SONNET] Nearest Neighbor Search with Early Termination\n");
		if(argc != 5){
            printf("!ERROR(main.cpp):Inappropriate parameters.\n");
			exit(0); 
	   }
	}else{
		printf("!ERROR: Please choose a algorithm\n");
		exit(0);
	}

	K_NN::ALG = alg;
	vector<K_NN*> imgs;
	TIMER_T runtime[4];
    TIMER_READ(runtime[0]);
    csv_parser csv;
    csv.init(csv_file);
    csv.set_enclosed_char('"', ENCLOSURE_OPTIONAL);
    csv.set_field_term_char(',');
    csv.set_line_term_char('\n');
    csv.get_row();
	printf("-- LOADING DATA ......\n");
    while (csv.has_more_rows()) {
        const char *csv_name = csv.get_row()[1].c_str();
        K_NN *img = new K_NN();
        snprintf(img->name, 50, format, csv_name);
        img->readImg();
        imgs.push_back(img);

    }
    TIMER_READ(runtime[1]);
	printf("DONE LOAD: %f s\n", TIMER_DIFF_SECONDS(runtime[0], runtime[1]));

#ifdef  EXACT
	printf("-- EXACT: ON\nSEARCHING FOR EXACT ......");
    int* index = new int[imgs[1]->rows * KNN];
    float* dist = new float[imgs[1]->rows * KNN];
    K_NN::ALG = 0;
    imgs[0]->buildIndex();
    imgs[0]->match(*imgs[1], index, dist, "w", MAX_THREADS);
    delete index;
    delete dist;
    printf("DONE EXACT: Found and saved exact results for %s\n", imgs[0]->name);
#else
	printf("-- INDEX BUILDING ......\n");
    // prepare to build index...
    switch (alg){
    case 1:
        // TYPE:    0-read saved; 
        //			1-svd appointed dimensions; 
        //			2 - svd percentage dimensions; 
        //			3 - svd percentage variation
        PCAIndex::TYPE = atoi(argv[4]);
        PCAIndex::VALUE = atof(argv[5]);
		printf("Type:%d\nValue:%f\n", PCAIndex::TYPE, PCAIndex::VALUE);
        break;
    case 2:
        // TREES:	0 - read saved; 
        //			!0 - number of trees to build
        RKDIndex::TREES = atoi(argv[4]);
		printf("Number of tress:%d\n", RKDIndex::TREES);
        break;
    case 3:
        SCFIndex::SUBSPACES = atoi(argv[4]);
        SCFIndex::CLUSTERS = atoi(argv[5]);
        printf("Number of subspaces:%d\nNumber of clusters:%d\n", SCFIndex::SUBSPACES, SCFIndex::SUBSPACES);
        break;
    case 4:
        RBCIndex::REP = atoi(argv[4]);
        printf("Number of replica:%d\n", RBCIndex::REP);
        break;
    default:
        break;
    }
    K_NN::ALG = alg;
	TIMER_READ(runtime[1]);
    imgs[0]->buildIndex();
	TIMER_READ(runtime[2]);
	printf("DONE BUILD: %f s\n", TIMER_DIFF_SECONDS(runtime[1], runtime[2]));

	printf("-- SEARCH USING MAX THREADS WITH PRECISION & FILTERING STATISTICS ......\n");
    //prepare to search...
    switch (alg){
		case 1:
            PCAIndex::HEAP = atoi(argv[6]);
            PCAIndex::BLOCK = atoi(argv[7]);
			imgs[0]->prepare(*imgs[1]);
			printf("[m] Heap scaling factor:%s\n", argv[6]);
			printf("[S] Number of partitions:%s\n", argv[7]);
            break;
		case 2:
			RKDIndex::CHECKS = atoi(argv[5]);
			printf("Number of checks:%s\n", argv[5]);
			break;
	    case 3:
		    SCFIndex::SCALE_U = atof(argv[6]);
			SCFIndex::SCALE_D = atof(argv[7]);
			printf("Upper scaling factor:%s\n", argv[6]);
			printf("Lower scaling factor:%s\n", argv[7]);
			break;
		case 5:
			SONNETIndex::FRAC = atof(argv[4]);
			printf("[FRAC] Fraction:%s\n", argv[4]);
			break;
		default:
			break;
    }

    int* index = new int[imgs[1]->rows * KNN];
    float* dist = new float[imgs[1]->rows * KNN];
    
    if(alg == 5){
        TIMER_READ(runtime[2]);
        imgs[0]->match(*imgs[1], index, dist, "o", MAX_THREADS);
        TIMER_READ(runtime[3]);
        float precision = imgs[0]->precisionWithDist(imgs[1]->data, imgs[1]->rows, index);
        printf("Precision: %f \%\n", precision);
		printf("Time cost: %f s\n", TIMER_DIFF_SECONDS(runtime[2], runtime[3]));
    }else{
		TIMER_READ(runtime[2]);
		imgs[0]->match(*imgs[1], index, dist, "f", MAX_THREADS);
		TIMER_READ(runtime[3]);
		float precision = imgs[0]->precision(imgs[1]->rows, index);
		printf("precision: %f \%\n", precision);
		printf("Time cost: %f s\n", TIMER_DIFF_SECONDS(runtime[2], runtime[3]));
    }
    
	uint64_t start, end, time = 0;
    int skip = SKIP, i;
	printf("-- SEARCH USING INCREASING NUMBER OF THREADS STARTING FROM 1 ......\n");
    printf("NOTE: please set appropriate SKIP and MAX_THREADS for searching.\n");
	printf("Skip:%d\n", skip);
	printf("Max number of threads:%d\n", MAX_THREADS);
	printf("Search time in seconds(s):\n");
	for (int threads = 1; threads <= MAX_THREADS; threads += skip) {
        TIMER_READ(runtime[2]);
        imgs[0]->match(*imgs[1], index, dist, "o", threads);
        if (threads == 1) {
            threads = 0;
        }
        TIMER_READ(runtime[3]);
        fflush(0);
        printf("%f\t", TIMER_DIFF_SECONDS(runtime[2], runtime[3]));
    }
    printf("\nDONE SEARCH.\n==============================\n");
    delete index;
    delete dist;
#endif
    exit(0);
}

