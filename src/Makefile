#HOST=CPU
CC=icc
FLAGS= -O3 -w -fopenmp
LOCATION=..
LDFLAGS=-L$(LOCATION)/lib -lflann -lgsl -lrbc
INCLUDE=-I$(LOCATION)/include

all: k_nn.o csv_parser.o common.o bfs_index.o pca_index.o sonnet_index.o rkd_index.o scf_index.o rbc_index.o main.o 
	$(CC) $(FLAGS) $(LDFLAGS) csv_parser.o common.o bfs_index.o pca_index.o sonnet_index.o rkd_index.o scf_index.o rbc_index.o k_nn.o main.o -o main

csv_parser.o: csv_parser.cpp csv_parser.h
	$(CC) $(FLAGS) $(INCLUDE) -c csv_parser.cpp

common.o: common.cpp common.h
	$(CC) -vec-report1 $(FLAGS) $(INCLUDE) -c common.cpp

bfs_index.o: bfs_index.cpp bfs_index.h bfs_linear_index.h common.h
	$(CC) $(FLAGS) $(INCLUDE) -c bfs_index.cpp

pca_index.o: pca_index.cpp pca_index.h pca_svd_index.h common.h
	$(CC) $(FLAGS) $(INCLUDE) -c pca_index.cpp

rkd_index.o: rkd_index.cpp rkd_index.h rkd_kdtree_index.h common.h
	$(CC) $(FLAGS) $(INCLUDE) -c rkd_index.cpp

scf_index.o: scf_index.cpp scf_index.h scf_kmeans_index.h common.h
	$(CC) $(FLAGS) $(INCLUDE) -c scf_index.cpp

rbc_index.o: rbc_index.cpp rbc_index.h common.h
	$(CC) $(FLAGS) $(INCLUDE) -c rbc_index.cpp

sonnet_index.o: sonnet_index.cpp sonnet_index.h common.h
	$(CC) $(FLAGS) $(INCLUDE) -c sonnet_index.cpp

k_nn.o: k_nn.cpp k_nn.h bfs_index.h pca_index.h sonnet_index.h rkd_index.h scf_index.h rbc_index.h common.h
	$(CC) $(FLAGS) $(INCLUDE) -c k_nn.cpp

main.o: main.cpp csv_parser.h k_nn.h common.h
	$(CC) $(FLAGS) $(INCLUDE) -c main.cpp

clean:
	rm -f *.o
	rm -f main
