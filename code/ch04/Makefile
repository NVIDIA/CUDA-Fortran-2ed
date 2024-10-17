OBJS = twoKernels pipeline streamSync eventSync \
       defaultStream defaultStreamVar defaultStreamVarExplicit \
       differentStreamTypes concurrentKernels \
       sharedExample sharedMultiple syncthreads ballot shfl \
       raceAndAtomic raceAndAtomicShared threadfence cgReverse \
       smooth swap

# section 4.1.2.1

twoKernels: twoKernels.cuf
	nvfortran -o $@ $<

# section 4.1.3

pipeline: pipeline.cuf
	nvfortran -o $@ $<

# section 4.1.4.2

streamSync: streamSync.cuf
	nvfortran -o $@ $<

# section 4.1.4.3

eventSync: eventSync.cuf
	nvfortran -o $@ $<

# section 4.1.5.1

defaultStream: defaultStream.cuf
	nvfortran -o $@ $<

defaultStreamVar: defaultStreamVar.cuf
	nvfortran -o $@ $<

defaultStreamVarExplicit: defaultStreamVarExplicit.cuf
	nvfortran -o $@ $<

# section 4.1.5.2

differentStreamTypes: differentStreamTypes.cuf
	nvfortran -o $@ $<

concurrentKernels: concurrentKernels.cuf
	nvfortran -o $@ $<

# section 4.2.1

sharedExample: sharedExample.cuf
	nvfortran -o $@ $<

sharedMultiple: sharedMultiple.cuf
	nvfortran -o $@ $<

# section 4.2.2

syncthreads: syncthreads.cuf
	nvfortran -o $@ $<

# section 4.2.3

ballot: ballot.cuf
	nvfortran -o $@ $<

# section 4.2.3.1

shfl: shfl.cuf
	nvfortran -o $@ $<

# section 4.2.4

raceAndAtomic: raceAndAtomic.cuf
	nvfortran -o $@ $<

raceAndAtomicShared: raceAndAtomicShared.cuf
	nvfortran -o $@ $<

# section 4.2.5

threadfence: threadfence.cuf
	nvfortran -o $@ $<

# section 4.2.6

cgReverse: cgReverse.cuf
	nvfortran -o $@ $<

# section 4.2.6.1

smooth: smooth.cuf
	nvfortran -o $@ $<

# section 4.2.6.2

swap: swap.cuf
	nvfortran -o $@ -gpu=cc90 $<


clean:
	rm -rf $(OBJS) *.o *.mod *~
