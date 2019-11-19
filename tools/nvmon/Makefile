CFLAGS := -I${CUDA_HOME}/include -lnvidia-ml

all: nvmon

nvmon: nvmon.c
	$(CC) $(CFLAGS) $< -o $@
