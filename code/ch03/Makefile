OBJS = events multidim nvtxBasic nvtxAdv nvtxAdv2 nvtxAuto \
	limitingFactor peakBandwidth effectiveBandwidth

# section 3.1.2

events: events.cuf
	nvfortran -o $@ $<

# section 3.1.3

multidim: multidim.cuf
	nvfortran -o $@ $<

# section 3.1.4.1

nvtxBasic: nvtxBasic.cuf
	nvfortran -o $@ $< -cudalib=nvtx

# section 3.1.4.2

nvtxAdv: nvtxAdv.cuf
	nvfortran -o $@ $< -cudalib=nvtx

nvtxAdv2: nvtxAdv2.cuf
	nvfortran -o $@ $< -cudalib=nvtx

# section 3.1.4.3

nvtxAuto: nvtxAuto.cuf
	nvfortran -o $@ $< -Minstrument -traceback -cudalib=nvtx

# section 3.2

limitingFactor: limitingFactor.cuf
	nvfortran -o $@ $<

# section 3.3.1

peakBandwidth: peakBandwidth.cuf
	nvfortran -o $@ $<

# section 3.3.2

effectiveBandwidth: effectiveBandwidth.cuf
	nvfortran -o $@ $<

clean:
	rm -rf $(OBJS) *.o *.mod *~
