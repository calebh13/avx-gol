#!/bin/bash

N_VALUES=(4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
P_VALUES=(1 2 4 8 16 32)

for p in "${P_VALUES[@]}"; do
    for n in "${N_VALUES[@]}"; do

        [ "$n" -le "$p" ] && continue

        if [ "$p" -le 16 ]; then
            NODES=1
        else
            NODES=2
        fi

        ./scripts/.run_job.sh ${NODES} ${p} ./main ${n} 15

    done
done
