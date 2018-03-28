#!/usr/bin/env python3
import os

telamon_root = os.path.realpath("../../")
tuning_path = os.path.realpath(".")
setting_path = tuning_path + "/settings/"


key_list = ["num_worker", "log_file",  "algorithm", "stop_bound",
        "timeout", "delta", "distance_to_best", "monte_carlo","threshold",
        "new_nodes_order", "old_nodes_order"]
search_list = ["bound_order", "bandit"]
new_order_list = ["api", "random", "bound", "weighted_random"]
old_order_list = ["bound", "bandit", "weighted_random"]
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
    if key in ["delta", "stop_bound", "distance_to_best"]:
        try:
            t = float(val)
            return True
        except:
            print("Key {} should get an float, got {}".format(key, val))
            return False
    return ((key == "algorithm" and val in search_list) 
        or (key == "new_nodes_order" and val in new_nodes_order)
        or (key == "old_nodes_order" and val in old_nodes_order)
        or (key == "monte_carlo" and val in bool_list))


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

def clear_directory(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

filename = "test_py.log"
opts = dict()
opts["num_worker"] = 24
opts["algorithm"] = "bandit"
opts["timeout"] = 150
opts["monte_carlo"] = "true"

if __name__ == "__main__":
    if not os.path.exists(setting_path):
        os.makedirs(setting_path)
    clear_directory(setting_path)
    for i in range(8):
        opts["delta"] = pow(2, i) * 0.00001
        for j in range(1, 4):
            opts["threshold"] = j * 10
            filename = ("d" + "-" + "{:3e}".format(opts["delta"]) + "_" + "t" 
                    + "{}".format(opts["threshold"]) +".toml")
            create_setting_file(opts, setting_path + filename)
