CC_FLAGS := -I${CUDA_HOME}/include -lnvidia-ml

all: nvmon

.PHONY: all

nvmon: nvmon.c
	$(CC) $(CC_FLAGS) $< -o $@
