# PCAF: Pricipal Conponent Analysis based Filtering

## What's PCAF
PCAF is an efficient algorithm for approximate k nearest neighbors (kNN) search.
It achieves high precision, speed and parallel scalability.

### How to intall
PCAF depends on [FLANN](http://www.cs.ubc.ca/research/flann/) and [Eigen](http://eigen.tuxfamily.org/dox/GettingStarted.html).  
So, first please make sure FLANN and Eigen is ready to use.
Below is an example of how FLANN can be compiled on Linux (replace x.y.z with the corresponding version number).

	$ cd flann-x.y.z-src
	$ mkdir build
	$ cd build
	$ cmake ..
	$ make
For Eigen, you just need to placed Eigen's source code in the include path.
With GCC/ICC, use the -I option to achieve find the header file.
There is a ```Makefile``` in PCAF, you can simple run ```make``` to get a executable file ```main```.

## How to use it

To search with PCAF, you need to specify 

* the executable file: ``` ./main```
* the format for reading a csv file, which contains the binary data file (optdigits.tra.bin) and binary query file (optdigits.tes). ``` ../dataset/%s ../dataset/optdigits.csv```
* Type and coresponding value.  
PCAF provides three ways to set the dimensionality of PCA space.
	* 1: directly appoint the number of dimension, $d$;
	* 2: the number of dimension in percentage;  
	Like if the input data is 64-dimensional points, and 50% of the dimensions is desired.
	Then set the value to be 0.5, and PCAF will use $d=64\times 0.5=32$ dimensions in PCA space.
	* 3: percentage of variation remained in the PCA space.  
	It will calculate the dimensionality of PCA space itself, we recommend the value to be 0.9, which means 90% of the variation in the data will be remained in PCA space.  
* Heap scaling factor ($m$).
* Number of data partitions ($S$).  

```
Params:
	TYPE	0: read saved index
			1: appointed dims
			2: appointed percentage of dims
			3: appointed variation
	VALUE	[float | 0.0 - 1.0] 		values of TYPE
	HEAP	[int 	| usually 1 - 5] 	heap scaling factor
	BLOCK	[int 	| usually 1 - 16] 	partitions
Sample command to run:
	./main ../dataset/%s ../dataset/optdigits.csv pcaf 1 8 2 4
```

There are some defines might to be worried:  
* In ```k_nn.h```:  
```#define KNN 2``` --- set the value of k-NN
	
* In ```main.cpp```:  
```#define SKIP 4``` --- set the step of increasing number of threads;  ```#define MAX_THREADS 16``` --- the maximum number of threads to run, better to be less than the cores in your machine.  
```#define EXACT``` --- enable this define, and make the executable file for generating true k-NN results, which will be save for verification for other runs.

### Other algorithms supports
 
This repository also provides several other kNN algorithms, they are:

* Brute Force (bfs)  
Produce exact kNN results provided by FLANN.
* Randomized KD Trees (RKD)  
provided by FLANN.  

```
Params:  
	TREES	[int] number of trees
	CHECKS	[int] number of checks
Sample command to run:
	./main ../dataset/%s ../dataset/optdigits.csv rkd 4 8
```
* Random Ball Cover (RBC)  
Reference: L. Cayton, “Accelerating nearest neighbor search on manycore systems,” in Parallel & Distributed Processing Symposium (IPDPS), 2012 IEEE 26th International. IEEE, 2012, pp. 402–413.  
To use RBC, you need to include rbc lib, see the ```Makefile```

```
Params:  
	REP		[int] replica, number of representatives = REP*sqrt(N)
Sample command to run:
	./main ../dataset/%s ../dataset/optdigits.csv rbc 4
```
* Subspace Clusters for Filtering (SCF)  
Reference: X. Tang, Z. Huang, D. Eyers, S. Mills, and M. Guo, “Scalable multi- core k-NN search via Subspace Clustering for Filtering,” Parallel and Distributed Systems, TPDS, IEEE Transactions on, vol. 26, no. 12, pp. 3449–3460, 2015.  
Here, we greatly thank the author Tang for the sharing.  

```
Params:  
	SUBSPACE	[int] subspaces
	CLUSTER		[int] clusters for each subspace
	SCALE_U		[float | 0.0 - 1.0] positive scale factor
	SCALE_D		[float | 0.0 - 1.0] negtive scale factor
Sample command to run:
	./main ../dataset/%s ../dataset/optdigits.csv scf 8 16 0 0
```
* Nearest Neighbor Search with Early Termination (SONNET)  
Reference: M. Al Hasan, H. Yildirim, and A. Chakraborty, “SONNET: Efficient approximate nearest neighbor using multi-core,” in Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 2010, pp. 719–724.

```
Params:
	FRAC		[float | 0.0 - 1.0] faction of points for search
Sample:
	./main ../dataset/%s ../dataset/optdigits.csv sonnet 0.05
```

### Raw data of experimental results
This reponsitory also stores the raw data of the PCAF experimental results. 
This PCAF project is conducted by Huan FENG, and the achievement has been submitted to ICPP 2016 and extension to TC 2017. 
The raw data belongs to Tsinghua University and University of Otago. The reason for its existence on github is for reviewers' convenience to check.