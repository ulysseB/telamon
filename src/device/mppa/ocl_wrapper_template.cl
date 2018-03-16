void {name}({cl_arg_defs}__global void* __timer_ptr);

__kernel void wrapper({cl_arg_defs}__global void* __timer_ptr) {{
  {name}({arg_names}__timer_ptr);
}}
