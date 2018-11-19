#!/bin/bash

set -euo

ORDERS=(
	"lower_layout,size,dim_kind,dim_map,mem_space,order,inst_flag"
)
#	"lower_layout,dim_kind,size,dim_map,mem_space,order,inst_flag"

KERNELS=(
	"--batchmm 512 32 32 64 --bmm-static-sizes --bmm-reuse-b --cut 143000"
	"--batchmm 512 32 32 64 --bmm-static-sizes --cut 177000"
)
# "--matmul 256 256 32 --no-matmul-fixed-tiling --cut 21400"
#"--cut 7050000"
#	"--matmul 1024 1024 1024 --no-matmul-fixed-tiling --cut 4810000"
#	"--matmul 1024 1024 1024 --no-matmul-fixed-tiling --matmul-stride-a 32 --cut 10100000"

ARGS=(
	"--max-cut-depth 10"
)
#	"--matmul 1024 1024 1024 --no-matmul-fixed-tiling --max-depth 10"
#	"--matmul 1024 1024 1024 --no-matmul-fixed-tiling --max-cut-depth 10 --cut 4810000 --max-depth 10"
#	"--matmul 1024 1024 1024 --no-matmul-fixed-tiling --max-cut-depth 10 --cut 4810000"
#
# "--matmul 256 256 32 --no-matmul-fixed-tiling"
#	"--matmul 256 256 32 --no-matmul-fixed-tiling --max-cut-depth 10 --cut "
#	"--max-cut-depth 10 --cut "
#)

for order_ix in ${!ORDERS[*]}; do
	order=${ORDERS[$order_ix]};
	for arg_ix in ${!ARGS[*]}; do
		arg=${ARGS[$arg_ix]};
		for kernel_ix in ${!KERNELS[*]}; do
			kernel=${KERNELS[$kernel_ix]};

			NAME=$(cat /proc/sys/kernel/random/uuid)
			NAME=${NAME##*-}
			echo "Running $NAME..."
			echo /home/elarnon/code/telamon/target/release/treesize2 \
				--num-playouts 100000 --evaluator policy \
				--output "experiments_size2/$NAME" \
				$arg \
				--ordering "$order" \
				$kernel \
				|| true
			/home/elarnon/code/telamon/target/release/treesize2 \
				--num-playouts 100000 --evaluator policy \
				--output "experiments_size2/$NAME" \
				$arg \
				--ordering "$order" \
				$kernel \
				|| true
		done
	done
done
