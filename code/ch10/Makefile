OBJS = finiteDifference finiteDifferenceStr limitingFactor laplace2D

# section 10.1.*

finiteDifference: precision_m.cuf derivative_m.cuf finiteDifference.cuf 
	nvfortran -o $@ -O3 -gpu=ptxinfo $^

# section 10.1.2

limitingFactor: precision_m.cuf limitingFactor.cuf
	nvfortran -o $@ -O3 $^

# section 10.1.4

finiteDifferenceStr: precision_m.cuf derivativeStr_m.cuf finiteDifferenceStr.cuf
	nvfortran -o $@ -O3 $^

# section 10.2

laplace2D: laplace2D.cuf 
	nvfortran -o $@ -O3 $^


clean:
	rm -rf $(OBJS) *.o *.mod *~
