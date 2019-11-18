#!/bin/bash
# Machine configuration script for GPU benchmarking.
#
# This will set the CPU governor to `performance`, enable persistence mode on
# the GPU (otherwise all other settings that we can make get erased immediately
# because that is definitely intuitive) and sets the application clocks on the
# GPU to their default values.
#
# It needs to be run before starting the benchmarks.

set -euo pipefail

echo "Setting cpu governor to performance..."
cpufreq-info -o | \
	sed -n 's/^CPU\s*\([0-9]\+\).*/\1/p' | \
	xargs -i -n 1 sudo cpufreq-set -c {} -g performance
echo "OK."

echo "Enabling persistence mode..."
sudo nvidia-smi -pm 1
echo "OK."

echo "Setting application clocks..."
nvidia-smi -q -d CLOCK | \
	grep -A2 'Default Applications Clocks' | \
	tail -n2 | cut -d':' -f2 | cut -d' ' -f2 | tac | paste -d, -s | \
	xargs -i sudo nvidia-smi -ac {}
echo "OK."
