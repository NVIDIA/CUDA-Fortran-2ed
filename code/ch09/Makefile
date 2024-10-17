OBJS = rng_gpu_sp rng_gpu_dp pi_sp pi_dp ieee_accuracy \
       pi_performance_sp shflExample testPiGridGroup \
       accuracy_sum mc_european_single mc_european_double


# section 9.1

rng_gpu_sp: precision_m.F90 generate_randomnumbers.cuf
	nvfortran -O3 -o $@ $^ -cudalib=curand

rng_gpu_dp: precision_m.F90 generate_randomnumbers.cuf
	nvfortran -O3 -DDOUBLE -o $@ $^ -cudalib=curand

# section 9.2

pi_sp: precision_m.F90 compute_pi.cuf
	nvfortran -O3 -o $@ $^ -cudalib=curand

pi_dp: precision_m.F90 compute_pi.cuf
	nvfortran -DDOUBLE -O3 -o $@ $^ -cudalib=curand

# section 9.2.1

ieee_accuracy: ieee_accuracy.f90
	nvfortran -o $@ $^

# section 9.3.*

pi_performance_sp: precision_m.F90 pi_shared.cuf pi_shfl.cuf \
		   pi_lock.cuf pi_gridGroup.cuf compute_pi_performance.CUF
	nvfortran -O3 -DLOOP -o $@ $^ -cudalib=curand

# section 9.3.1

shflExample: shflExample.cuf
	nvfortran -o $@ $<

# section 9.3.3

testPiGridGroup: precision_m.F90 pi_gridGroup.cuf testPiGridGroup.cuf
	nvfortran -o $@ $^ -cudalib=curand

# section 9.4

accuracy_sum: accuracy_sum.f90
	nvfortran -o $@ $^

# section 9.5

mc_european_single: precision_m.F90 montecarlo_european_option.cuf 
	nvfortran -O3 -Minfo -o $@ $^ -cudalib=curand

mc_european_double: precision_m.F90 montecarlo_european_option.cuf 
	nvfortran -O3 -Minfo -DDOUBLE -o $@ $^ -cudalib=curand

clean:
	rm -rf $(OBJS) *.o *~ *.mod *~

