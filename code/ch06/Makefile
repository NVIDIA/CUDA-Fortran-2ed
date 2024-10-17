OBJS = portingBase portingManaged portingDevice \
       portingManaged_HOST portingManaged_CUDA \
       portingDevice_HOST portingDevice_CUDA \
       portingManagedSent_HOST portingManagedSent_CUDA \
       portingDeviceSent_HOST portingDeviceSent_CUDA \
       laplace2D laplace2DUse_HOST laplace2DUse_CUDA \
       portingAssociate_HOST portingAssociate_CUDA \
       laplace2DAssoc_HOST laplace2DAssoc_CUDA test_union \
       test_union testCompact testCompactOpt

# section 6.1

portingBase: portingBase.f90
	nvfortran -o $@ $<

portingManaged: portingManaged.cuf
	nvfortran -o $@ $<

portingDevice: portingDevice.cuf
	nvfortran -o $@ $<

# section 6.2

portingManaged_HOST: portingManaged_CUDA.F90
	nvfortran -o $@ $<

portingManaged_CUDA: portingManaged_CUDA.F90
	nvfortran -cuda -o $@ $<

portingDevice_HOST: portingDevice_CUDA.F90
	nvfortran -o $@ $<

portingDevice_CUDA: portingDevice_CUDA.F90
	nvfortran -cuda -o $@ $<


portingManagedSent_HOST: portingManagedSent.f90
	nvfortran -o $@ $<

portingManagedSent_CUDA: portingManagedSent.f90
	nvfortran -cuda -o $@ $<

portingDeviceSent_HOST: portingDeviceSent.F90
	nvfortran -o $@ $<

portingDeviceSent_CUDA: portingDeviceSent.F90
	nvfortran -cuda -o $@ $<

# section 6.3.1

laplace2D: laplace2D.f90
	nvfortran -o $@ $<

laplace2DUse_HOST: laplace2DUse.F90
	nvfortran -O3 -o $@ $<

laplace2DUse_CUDA: laplace2DUse.F90
	nvfortran -O3 -cuda -o $@ $<

# section 6.3.2

portingAssociate_HOST: portingAssociate.f90
	nvfortran -o $@ $<

portingAssociate_CUDA: portingAssociate.f90
	nvfortran -cuda -o $@ $<

laplace2DAssoc_HOST: laplace2DAssoc.f90
	nvfortran -O3 -o $@ $<

laplace2DAssoc_CUDA: laplace2DAssoc.f90
	nvfortran -O3 -cuda -o $@ $<

# section 6.4

test_union: union_m.cuf test_union.cuf
	nvfortran -o $@ $^

# section 6.5

testCompact: compact_m.cuf testCompact.cuf
	nvfortran -o $@ $^

testCompactOpt: compactOpt_m.cuf testCompact.cuf
	nvfortran -o $@ $^


clean:
	rm -rf $(OBJS) *.o *.mod *~
