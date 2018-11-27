#!/bin/bash

set -euo

ORDERS=(
	"dim_kind,size,lower_layout,inst_flag,order,mem_space,dim_map"
	"lower_layout,size,dim_kind,dim_map,mem_space,order,inst_flag"
)
# "lower_layout,size,dim_kind,dim_map,mem_space,order,inst_flag"
#	"lower_layout,dim_kind,size,dim_map,mem_space,order,inst_flag"

KERNELS=(
	"--matmul 1024 1024 1024 --no-matmul-fixed-tiling --max-depth 9"
)
#
# "--matmul 256 256 32 --no-matmul-fixed-tiling"
#	"--matmul 256 256 32 --no-matmul-fixed-tiling --max-cut-depth 10 --cut "
#	"--max-cut-depth 10 --cut "
#)

ARGS=(
	"--num-playouts 100000 --evaluator policy"
)

for order_ix in ${!ORDERS[*]}; do
	order=${ORDERS[$order_ix]};
	for kernel_ix in ${!KERNELS[*]}; do
		kernel=${KERNELS[$kernel_ix]};
		for arg_ix in ${!ARGS[*]}; do
			arg=${ARGS[$arg_ix]};

			NAME=$(cat /proc/sys/kernel/random/uuid)
			NAME=${NAME##*-}
			echo "Running $NAME..."
			/home/elarnon/code/telamon/target/release/treesize2 \
				--output "experiments_nicocheck/$NAME" \
				$arg \
				--ordering "$order" \
				$kernel \
				|| true
		done
	done
done
