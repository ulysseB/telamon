void {name}({cl_arg_defs});

__kernel void wrapper({cl_arg_defs}) {{
  {name}({arg_names});
}}
