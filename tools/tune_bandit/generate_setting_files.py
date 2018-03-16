#!/usr/bin/env python3

setting_path = "/home/nicolas/telamon/settings/"


key_list = ["num_worker", "filename", "threshold", "search_type",
    "node_selection", "expanded_selection", "stop_search", "timing",
    "timeout", "delta", "bound"]
search_list = ["standard", "statistic"]
node_list = ["standard", "random", "best", "mixed"]
expanded_list = ["standard", "bandit", "mixed"]
bool_list = ["true", "false"]



def create_setting_file(options_dict, filename):
  if not check_dict(options_dict):
    print("Invalid Options dict")
    return
  formatted_opts = add_quotes(options_dict)
  with open(filename, 'w+') as f:
    for key, value in formatted_opts.items():
      f.write("{} = {}\n".format(key, value))

def check_tuple(key, val):
  if key in ["num_worker", "threshold", "timeout"]:
    try:
      t = int(val)
      return True
    except:
      print("Key {} should get an int, got {}".format(key, val))
      return False
  if key in ["delta", "bound"]:
    try:
      t = float(val)
      return True
    except:
      print("Key {} should get an float, got {}".format(key, val))
      return False
  return ((key == "search_type" and val in search_list) 
      or (key == "node_selection" and val in node_list)
      or (key == "expanded_selection" and val in expanded_list)
      or (key == "stop_search" and val in bool_list)
      or (key == "timing" and val in bool_list))


def check_dict(options_dict):
  assert all(key in key_list for key in options_dict)
  for key, value in options_dict.items():
    if not check_tuple(key, value):
      print("Key {} is wrong in config file".format(key))
      return False
  return True

def add_quotes(opt_dicts):
  new_opts = dict()
  for key, value in opt_dicts.items():
    new_val = '"' + "{}".format(value) + '"'
    new_opts[key] = new_val
  return new_opts


filename = "test_py.log"
opts = dict()
opts["num_worker"] = 24
opts["search_type"] = "statistic"
opts["timing"] = "true"

for t in range(1, 11):
  opts["timeout"] = t
  for i in range(8):
    opts["delta"] = pow(2, i) * 0.00001
    for j in range(8):
      opts["threshold"] = pow(2, j)
      filename = ("{}".format(opts["timeout"]) + "-" + "m" + "_" + "d" + "-" +
          "{:3e}".format(opts["delta"]) + "_" + "t" + "{}".format(opts["threshold"])
          +".toml")
      create_setting_file(opts, setting_path + filename)
     
