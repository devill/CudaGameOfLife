
CC=g++
NVCC=/usr/local/cuda/bin/nvcc 
FLAGS=
HEADERS=$(wildcard *.h)
OBJ=$(HEADERS:.h=.o)
TESTS=$(wildcard *Test.cu)

gol: main.cu $(OBJ)
	$(NVCC) $(FLAGS) -o $@ main.cu $(OBJ) -lglut

test: $(TESTS:.cu=)

runtests: $(TESTS:.cu=)
	@ls *Test | while read i; do ./$$i; done

# compile tests
%Test: %Test.cu $(OBJ) $(HEADERS) $(wildcard *.hpp) $(wildcard *.cu)
	$(NVCC) $(FLAGS) -o $@ $*Test.cu $(OBJ) -lgtest_main -lpthread -lglut
	@./$*Test

# pull in dependency info for *existing* .o files
-include $(OBJ:.o=.d)

# compile and generate dependency info
%.o: %.cu %.h
	$(NVCC) $(FLAGS) -c $*.cu -o $*.o
	$(CC) -MM $*.cu > $*.d

clean:
	rm -f gol *Test *.o *.d 


