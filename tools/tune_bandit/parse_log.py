#!/usr/bin/env python3

import re
import os

log_path = "/home/nicolas/telamon/log/"

def parse_file(filename, delta_dict, threshold_dict):
  with open(filename, 'r') as f:
    for line in f:
      m = re.search('delta\s*:\s*(.*)', line)
      if m:
        delta = float(m.group(1))
      m = re.search('threshold\s*:\s*(.*)', line)
      if m:
        threshold = int(m.group(1))
      m = re.search('timeout\s*:\s*(.*)', line)
      if m:
        timeout = int(m.group(1))
      m = re.search("Complete.*(\d\.\d{4}e\d)", line)
      if m:
        score = float(m.group(1))
    if not timeout in delta_dict:
      delta_dict[timeout] = dict()
      threshold_dict[timeout] = dict()
    if not delta in delta_dict[timeout]:
      delta_dict[timeout][delta] = []
    if not threshold in threshold_dict[timeout]:
      threshold_dict[timeout][threshold] = []
    delta_dict[timeout][delta].append(score)
    threshold_dict[timeout][threshold].append(score)

def get_infos():
  delta_dict = dict()
  threshold_dict = dict()
  for filename in os.listdir(log_path):
    parse_file(log_path + filename, delta_dict, threshold_dict)
  for timeout in delta_dict:
    print("For timeout {}".format(timeout))
    for delta, score_list in delta_dict[timeout].items():
      mean_score = sum(score_list) / len(score_list)
      print("For delta = {}, mean score : {:.4e} on {} values".format(delta,
        mean_score, len(score_list)))
    for threshold, score_list in threshold_dict[timeout].items():
      mean_score = sum(score_list) / len(score_list)
      print("For threshold = {}, mean score : {:.4e} on {} values"
          .format(threshold, mean_score, len(score_list)))

get_infos()
