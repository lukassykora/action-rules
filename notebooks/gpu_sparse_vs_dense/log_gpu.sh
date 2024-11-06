#!/bin/bash
duration=180  # 3 minutes in seconds
end_time=$(( $(date +%s) + duration ))

while [ $(date +%s) -lt $end_time ]; do
    /home/jupyter-xsykl04@vse.cz/.local/bin/gpustat -cp >> $1
    sleep 1
done