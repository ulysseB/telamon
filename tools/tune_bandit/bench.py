#!/usr/bin/env python3

import os
import subprocess

tuning_path = os.path.dirname(os.path.realpath(__file__))
telamon_root = os.path.realpath(tuning_path + "/../../")
log_path = tuning_path + "/log/"
setting_path = tuning_path + "/settings/"


if __name__ == "__main__":
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    for filename in os.listdir(setting_path):
      copy_command = "cp {} {}/Settings.toml".format(setting_path + filename,
              telamon_root)
      process = subprocess.Popen(copy_command.split())
      output, error = process.communicate()
      for i in range(4):
        logname = filename.replace(".toml", ".log")
        logfile = log_path + logname + "."  + str(i)
        run_command = "cargo run --release --features=cuda --example=sgemm -- -j24 -f" + logfile
        subprocess.call(run_command.split())
