OBJS = HDtransfer sliceTransfer async \
       offset stride local constant constantAttribute \
       maxSharedMemory transpose \
       parallelism parallelismPipeline cufILP fma


# section 5.1.1

HDtransfer: HDtransfer.cuf
	nvfortran -o $@ $<

sliceTransfer: sliceTransfer.cuf
	nvfortran -o $@ $<

async: async.cuf
	nvfortran -o $@ $<

# section 5.2.2.1

assumedShapeSize: assumedShapeSize.cuf
	nvfortran -c -gpu=ptxinfo $<

# section 5.2.2.2

offset: offset.cuf
	nvfortran -o $@ $<

stride: stride.cuf
	nvfortran -o $@ $<

# section 5.2.3

local: local.cuf
	nvfortran -o $@ -gpu=keep $<

# section 5.2.4

constant: constant.cuf
	nvfortran -o $@ $<

constantAttribute: constantAttribute.cuf
	nvfortran -o $@ $<

# section 5.2.5

loads: loads.cuf
	nvfortran -c -gpu=keep $<

# section 5.2.6.1

maxSharedMemory: maxSharedMemory.cuf
	nvfortran -o $@ $<

# section 5.2.6.2

transpose: transpose.cuf
	nvfortran -o $@ $<

# section 5.2.7

spill: spill.cuf
	nvfortran -c -gpu=ptxinfo $<

# section 5.3.1

parallelism: parallelism.cuf
	nvfortran -o $@ $<

# section 5.3.2.1

parallelismPipeline: parallelismPipeline.cuf
	nvfortran -gpu=cc80 -o $@ $<

# section 5.3.2.2

cufILP: cufILP.cuf
	nvfortran -o $@ $<

# section 5.4.1.4

fma: fma.cuf
	nvfortran -c -gpu=fma,keep  $<	


clean:
	rm -rf $(OBJS) *.o *.mod *~ *.bin *.gpu *.ptx *.fat
