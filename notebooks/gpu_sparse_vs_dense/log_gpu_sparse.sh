#!/bin/bash
duration=180  # 3 minutes in seconds
end_time=$(( $(date +%s) + duration ))

while [ $(date +%s) -lt $end_time ]; do
    gpustat -cp >> gpu_memory_sparse_matrix.log
    sleep 1
done