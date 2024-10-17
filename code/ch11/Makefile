OBJS = fft_test_c2c_sp fft_test_c2c_dp fft_test_r2c_sp fft_test_r2c_dp \
       spectral_sp spectral_dp exampleOverlapFFT ns2d_sp ns2d_dp

# section 11.1

fft_test_c2c_sp: precision_m.F90 fft_test_c2c.cuf
	nvfortran -o $@ $^ -cudalib=cufft

fft_test_c2c_dp: precision_m.F90 fft_test_c2c.cuf
	nvfortran -DDOUBLE -o $@ $^ -cudalib=cufft

fft_test_r2c_sp: precision_m.F90 fft_test_r2c.cuf
	nvfortran -o $@ $^ -cudalib=cufft

fft_test_r2c_dp: precision_m.F90 fft_test_r2c.cuf
	nvfortran -DDOUBLE -o $@ $^ -cudalib=cufft

# section 11.2

spectral_sp: precision_m.F90 fft_derivative.cuf
	nvfortran -O2 -o $@ $^ -cudalib=cufft

spectral_dp: precision_m.F90 fft_derivative.cuf
	nvfortran -DDOUBLE -O2 -o $@ $^ -cudalib=cufft

# section 11.3

exampleOverlapFFT: precision_m.F90 exampleOverlapFFT_nvtx.cuf
	nvfortran -O3 -Minfo -Mpreprocess -o $@ $^ -cudalib=nvtx,cufft

# section 11.4

ns2d_sp: ns2d.cuf
	nvfortran -DSINGLE -O3 -Mpreprocess -o $@ $< -cudalib=cufft

ns2d_dp: ns2d.cuf
	nvfortran -O3 -Mpreprocess -o $@ $< -cudalib=cufft



clean:
	rm -rf $(OBJS) *.o *.mod *~ *.hdf

