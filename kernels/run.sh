#!/bin/bash

ORDERS=(
	"lower_layout,size,dim_kind,dim_map,mem_space,order,inst_flag"
	"lower_layout,dim_kind,size,dim_map,mem_space,order,inst_flag"
	"lower_layout,dim_kind,dim_map,size,mem_space,order,inst_flag"
	"lower_layout,dim_map,dim_kind,size,mem_space,order,inst_flag"
	"lower_layout,dim_map,size,dim_kind,mem_space,order,inst_flag"
	"size,lower_layout,dim_kind,dim_map,mem_space,order,inst_flag"
	"dim_kind,lower_layout,size,dim_map,mem_space,order,inst_flag"
	"size,dim_kind,dim_map,mem_space,order,inst_flag,lower_layout"
	"dim_kind,size,dim_map,mem_space,order,inst_flag,lower_layout"
)

KERNELS=(
	""
	"--matmul 256 256 32"
	"--matmul 1024 1024 1024"
)

ARGS=(
	"--num-playouts 100 --evaluator stratified --stratifier combined"
	"--num-playouts 100000 --evaluator policy"
)

for kernel_ix in ${!KERNELS[*]}; do
	kernel=${KERNELS[$kernel_ix]};
	for order_ix in ${!ORDERS[*]}; do
		order=${ORDERS[$order_id]};

		for arg_ix in ${!ARGS[*]}; do
			arg=${ARGS[$arg_ix]};

			NAME=$(cat /proc/sys/kernel/random/uuid)
			NAME=${NAME##*-}
			/home/elarnon/code/telamon/target/release/treesize2 \
				--output "experiments/$NAME" \
				$arg \
				--ordering "$order" \
				$kernel \
				|| true
		done
	done
done
