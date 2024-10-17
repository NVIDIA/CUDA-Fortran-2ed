OBJS = accuracy fma print debug memcheck initcheck

# section 2.1.1

accuracy: accuracy.cuf
	nvfortran -o $@ $<

# section 2.1.2

fma: fma.cuf
	nvfortran -c -gpu=keep $<

# section 2.2.1

print: print.cuf
	nvfortran -o $@ $<

# section 2.2.2

debug: debug.cuf
	nvfortran -o $@ -g -gpu=nordc $<

# section 2.2.3

memcheck: memcheck.cuf
	nvfortran -o $@ $<

initcheck: initcheck.cuf
	nvfortran -o $@ $<

clean:
	rm -rf $(OBJS) *.o *.mod *~ *.ptx *.gpu *.bin *.fat
