MCUDA = -cuda 
MCUDA_CURAND = $(MCUDA) -cudalib=curand

OBJS = 	ppmExampleHost ppmExampleCUDA \
	rayHost rayCUDA \
	sphereHost sphereCUDA \
	normalHost normalCUDA \
	twoSpheresHost twoSpheresCUDA \
	antialiasHost antialiasCUDA \
	diffuseHost diffuseCUDA \
	metalHost metalCUDA \
	dielectricHost dielectricCUDA \
	cameraHost cameraCUDA \
	defocusBlurHost defocusBlurCUDA \
	coverHost coverCUDA \
	triangleHost triangleCUDA \
	lightsHost lightsCUDA \
	textureHost textureCUDA

# section 12.1

ppmExampleHost: ppmExample.f90
	nvfortran -o $@ $^

ppmExampleCUDA: ppmExample.f90
	nvfortran -o $@ $(MCUDA) $^

# section 12.2

rgb_m.host.o: rgb_m.F90
	nvfortran -c -o $@ $^

rgb_m.cuda.o: rgb_m.F90
	nvfortran -c -o $@ $(MCUDA) $^

# section 12.3

rayHost: ray.F90 rgb_m.host.o
	nvfortran -o $@ $^

rayCUDA: ray.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA) $^

# section 12.4

sphereHost: sphere.F90 rgb_m.host.o
	nvfortran -o $@ $^

sphereCUDA: sphere.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA) $^

# section 12.5

normalHost: normal.F90 rgb_m.host.o
	nvfortran -o $@ $^

normalCUDA: normal.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA) $^


twoSpheresHost: twoSpheres.F90 rgb_m.host.o
	nvfortran -o $@ $^

twoSpheresCUDA: twoSpheres.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA) $^

# section 12.6

antialiasHost: antialias.F90 rgb_m.host.o
	nvfortran -o $@ $^

antialiasCUDA: antialias.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA_CURAND) $^

# section 12.7.1

diffuseHost: diffuse.F90 rgb_m.host.o
	nvfortran -o $@ $^

diffuseCUDA: diffuse.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA_CURAND) $^

# section 12.7.2

metalHost: metal.F90 rgb_m.host.o
	nvfortran -o $@ $^

metalCUDA: metal.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA_CURAND) $^

# section 12.7.3

dielectricHost: dielectric.F90 rgb_m.host.o
	nvfortran -o $@ $^

dielectricCUDA: dielectric.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA_CURAND) $^

# section 12.8

cameraHost: camera.F90 rgb_m.host.o
	nvfortran -o $@ $^

cameraCUDA: camera.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA_CURAND) $^

# section 12.9

defocusBlurHost: defocusBlur.F90 rgb_m.host.o
	nvfortran -o $@ $^

defocusBlurCUDA: defocusBlur.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA_CURAND) $^

# section 12.10

coverHost: cover.F90 rgb_m.host.o
	nvfortran -o $@ $^

coverCUDA: cover.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA_CURAND) $^

# section 12.11

triangleHost: triangle.F90 rgb_m.host.o
	nvfortran -o $@ $^

triangleCUDA: triangle.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA_CURAND) $^

# section 12.12

lightsHost: lights.F90 rgb_m.host.o
	nvfortran -o $@ $^

lightsCUDA: lights.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA_CURAND) $^

# section 12.13

textureHost: texture.F90 rgb_m.host.o
	nvfortran -o $@ $^

textureCUDA: texture.F90 rgb_m.cuda.o
	nvfortran -o $@ $(MCUDA_CURAND) $^



clean:
	rm -rf a.out *.mod *.o *~

clobber:
	rm -rf $(OBJS) *.ppm a.out *.mod *.o *~
