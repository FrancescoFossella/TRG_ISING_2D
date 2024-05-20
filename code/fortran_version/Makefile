FC = mpif90 
FFLAGS = -fbounds-check -fno-range-check #-Wno-missing-include-dirs
#DIRFLAGS=- I/usr/include/x86_64-linux-gnu/mpich/mpi.mod
target = run
sources = $(wildcard *.f90)
objs = $(sources:.f90=.o)
libs = -llapack -lblas 


all:
	$(FC) $(FFLAGS) $(DIRFLAGS) -c $(sources)
	$(FC) $(FFLAGS) $(DIRFLAGS) -o $(target) $(objs) $(libs)

clean: 
	rm -f *.o *.mod 
	rm -f *.parquet
	rm run

