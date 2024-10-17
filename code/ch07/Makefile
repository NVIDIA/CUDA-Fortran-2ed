OBJS = callingC sgemmLegacy sgemmNew getrfBatched gemmPerf \
       cusparseMV cusparseMV_Ex potr matmulTC matmulTranspose \
       cutensorContraction testSort

ARCH=sm_75
GPU=cc75

# section 7.1
# (Ubuntu 20.04 sytems require the -c++libs flag when linking)

c.o: c.cu
	nvcc -c -arch=$(ARCH) -rdc=true c.cu

callingC: callingC.cuf c.o
	nvfortran -gpu=$(GPU) -o $@ $^ -c++libs

# section 7.2.1

sgemmLegacy: sgemmLegacy.cuf
	nvfortran -o $@ $< -cudalib=cublas -lblas

# section 7.2.2

sgemmNew: sgemmNew.cuf
	nvfortran -o $@ $< -cudalib=cublas

# section 7.2.3

getrfBatched: getrfBatched.cuf
	nvfortran -o $@ $< -cudalib=cublas

# section 7.2.4

gemmPerf: gemmPerf.cuf
	nvfortran -O3 -o $@ $< -cudalib=cublas

# section 7.3

cusparseMV: cusparseMV.cuf
	nvfortran -o $@ $< -cudalib=cusparse


cusparseMV_Ex: cudaforEx.cuf cusparseEx.cuf cusparseMV_Ex.cuf
	nvfortran -o $@ $^ -cudalib=cusparse

# section 7.4

potr: cudaforEx.cuf potr.cuf
	nvfortran -o $@ $^ -cudalib=cusolver

# section 7.5

matmulTC: matmulTC.cuf
	nvfortran -o $@ $^ -cudalib=cublas,cutensor

matmulTranspose: matmulTranspose.cuf
	nvfortran -o $@ $^ -cudalib=cutensor

# section 7.5.1

cutensorContraction: cutensorContraction.cuf
	nvfortran -o $@ $< -cudalib=cutensor

# section 7.6

thrust.C.o: thrust.cu
	nvcc -arch=$(ARCH) -c -o $@ $<

testSort: thrust.cuf testSort.cuf thrust.C.o
	nvfortran -gpu=$(GPU) -o $@ $^ -c++libs




clean:
	rm -rf $(OBJS) *.o *.mod *~

