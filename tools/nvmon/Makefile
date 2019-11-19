CFLAGS := -I${CUDA_HOME}/include -lnvidia-ml

all: nvmon

nvmon: nvmon.c
ifndef CUDA_HOME
	$(error Please specify CUDA_HOME)
endif
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f nvmon
