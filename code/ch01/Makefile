OBJS = incrementf90 incrementcuf multiblock multidim explicitInterface \
       managed managedImplicit multidimCUF managedCUF \
       managedCUF2f90 managedCUF2cuf deviceQuery pciBusID \
       errorHandling syncError asyncError version

all: $(OBJS)

# section 1.3.1

incrementf90: increment.f90
	nvfortran -o $@ $<

incrementcuf: increment.cuf
	nvfortran -o $@ $<

# section 1.3.2

multiblock: multiblock.cuf 
	nvfortran -o $@ $<

# section 1.3.3

multidim: multidim.cuf
	nvfortran -o $@ $<

# section 1.3.4

explicitInterface: explicitInterface.cuf
	nvfortran -o $@ $<

# section 1.3.5

managed: managed.cuf
	nvfortran -o $@ $<

managedImplicit: managedImplicit.cuf
	nvfortran -o $@ -gpu=managed $<

# section 1.3.6

multidimCUF: multidimCUF.cuf
	nvfortran -o $@ $<

managedCUF: managedCUF.cuf
	nvfortran -o $@ $<

managedCUF2f90: managedCUF2.f90
	nvfortran -o $@ $<

managedCUF2cuf: managedCUF2.f90
	nvfortran -o $@ -cuda $<

# section 1.4

deviceQuery: deviceQuery.cuf
	nvfortran -o $@ $<

# section 1.4.1

pciBusID: pciBusID.cuf
	nvfortran -o $@ $<

# section 1.5

errorHandling: errorHandling.cuf
	nvfortran -o $@ $<

syncError: syncError.cuf
	nvfortran -o $@ $<

asyncError: asyncError.cuf
	nvfortran -o $@ $<

# section 1.7

version: version.cuf
	nvfortran -o $@ $<

clean:
	rm -rf $(OBJS) *.o *.mod *~
