#!/usr/bin/env python3

import os
import subprocess

log_path = "log/"
setting_path = "settings/"


for filename in os.listdir(setting_path):
  copy_command = "cp {} Settings.toml".format(setting_path + filename)
  process = subprocess.Popen(copy_command.split())
  output, error = process.communicate()
  for i in range(4):
    logname = filename.replace(".toml", ".log")
    logfile = log_path + logname + "."  + str(i)
    run_command = "cargo run --release --features=cuda --example=sgemm -- -j24 -f" + logfile
    #process = subprocess.Popen(run_command.split())
    subprocess.call(run_command.split())
    #output, error = process.communicate()
