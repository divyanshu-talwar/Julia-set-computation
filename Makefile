#Mac compilation flags
#CPPFLAGS=-Wno-deprecated-declarations -I. -I/opt/local/include -I${GLM_PATH}
#LDFLAGS= -lstdc++ -O3 -L/opt/local/lib -lIL -lILU

#Linux compilation flags
CPPFLAGS=-I./devil/include
LDFLAGS= -L./devil/lib -lm -lstdc++ -lIL -lILU

parallel_julia:
	nvcc -o parallel_julia julia_gpu.cu ${CPPFLAGS} ${LDFLAGS}

clean:
	-rm -f parallel_julia
