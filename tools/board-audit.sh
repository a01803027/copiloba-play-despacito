#!/bin/sh
# Read-only preflight; intentionally does not alter IRQs, boot args, or governors.
set -eu
uname -a
cat /proc/cmdline
cat /sys/devices/system/cpu/isolated
cat /sys/devices/system/cpu/online
cat /proc/asound/cards
cat /proc/interrupts
for cpu in 0 1; do
    path="/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor"
    if [ -r "$path" ]; then cat "$path"; fi
done
if command -v v4l2-ctl >/dev/null 2>&1; then
    v4l2-ctl --all
    v4l2-ctl --list-formats-ext
fi
