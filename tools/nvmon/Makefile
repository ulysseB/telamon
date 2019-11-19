CFLAGS := -I${CUDA_HOME}/include -lnvidia-ml

all: nvmon

.PHONY: all

nvmon: nvmon.c
	$(CC) $(CFLAGS) $< -o $@
