#!/bin/bash

NSYS_COMMAND="nsys profile -t cuda,nvtx \
                        --cuda-event-trace=false \
                        --cuda-graph-trace=node \
                        --wait all \
                        -o ${NSIGHT_PREFIX}_%q{RANK} \
                        --force-overwrite true \
                        --sample=none \
                        --capture-range=cudaProfilerApi \
                        --capture-range-end=stop  \
                        "

CMD="python -u a2a.py"

if [ -v NSIGHT_PREFIX ]; then
    CMD="${NSYS_COMMAND} ${CMD}"
fi
     
echo ${CMD}
eval ${CMD}
