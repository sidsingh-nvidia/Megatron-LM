#!/bin/bash
# Submit a server (server/launch_minf.sh, or server/launch_vllm.sh with FRAMEWORK=vllm) and its
# client sweep (server/client.sh) from a run config.
#
# Run from the Megatron-LM repo root:
#   inference-bench/run.sh inference-bench/configs/math/ultrav3_wide_ep.env
#   inference-bench/run.sh inference-bench/configs/math/superv3_narrow_ep.env GEMM_BACKEND=vllm BENCH="4 8"
#
# A config is a KEY=value file of launcher / client.sh knobs plus:
#   FRAMEWORK minf (Megatron Core dynamic inference) | vllm   (default: minf)
#   NODES  server nodes; multi-node servers also get --segment SEGMENT, keeping each SEGMENT
#          nodes in one NVLink domain for the NVLS / symmetric-memory dispatch
#   SEGMENT nodes that must share an NVLink domain (default: NODES; lower it when the NVLink
#          groups are smaller than the job, e.g. narrow EP, for faster scheduling)
#   QOS    server QoS                                  (default: SBATCH_QOS from cluster.env)
#   TIME   time limit of both jobs                     (default: 2:00:00)
# KEY=VALUE arguments after the config override it. EXP_NAME defaults to the config's name
# (with _vllm appended for FRAMEWORK=vllm).
# The client starts with the server (--dependency=after) and waits for it to come up; when
# the sweep ends it shuts the server down. Results land in the server's experiment directory.

set -euo pipefail
CONFIG=${1:?usage: inference-bench/run.sh <config> [KEY=VALUE ...]}
shift
source "${INFERENCE_BENCH_CONFIG:-inference-bench/cluster.env}"
set -a
source "${CONFIG}"
for kv in "$@"; do export "${kv}"; done
SUFFIX=; [ "${FRAMEWORK:-minf}" == "vllm" ] && SUFFIX=_vllm
EXP_NAME=${EXP_NAME:-$(basename "${CONFIG}" .env)${SUFFIX}}
set +a
: "${NODES:?set NODES in ${CONFIG}}" "${BENCH:?set BENCH in ${CONFIG}}"

SERVER_OPTS=(-N "${NODES}" -t "${TIME:-2:00:00}")
[ "${NODES}" -gt 1 ] && SERVER_OPTS+=(--segment "${SEGMENT:-${NODES}}")
[ -n "${QOS:-}" ] && SERVER_OPTS+=(-q "${QOS}")

# Without BENCH the server only serves; the sweep runs from the client's own node.
SERVER_JOB=$(env -u BENCH sbatch --parsable "${SERVER_OPTS[@]}" --export=ALL \
    inference-bench/server/launch_${FRAMEWORK:-minf}.sh)
CLIENT_JOB=$(sbatch --parsable -t "${TIME:-2:00:00}" --dependency=after:"${SERVER_JOB}" \
    --export=ALL,SERVER_JOB="${SERVER_JOB}" inference-bench/server/client.sh)
echo "${EXP_NAME}: server ${SERVER_JOB} (${NODES} nodes), client ${CLIENT_JOB}"
