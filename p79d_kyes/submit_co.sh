#!/bin/sh
#-------------------------------------------------------------------------------
# RADMC-3D run 2 (13CO + 12CO, randomised n(H2); radmc_co_v2.json).
# Array task i -> simulation i / NCHUNK, chunk i % NCHUNK; frames with
# Ms > ms_max_frame are skipped. Merge / re-observations are separate, small
# jobs (submit_co_post.sh) once the array has finished.
#
#   sh submit_co.sh                     # all sims
#   sh submit_co.sh 0-5                 # these array tasks
#   EXTRA="--frames 151" sh submit_co.sh 0      # test
#-------------------------------------------------------------------------------
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
CONFIG=sims_mach_grid.json
RT=radmc_co_v2.json
NCHUNK=${NCHUNK:-6}
NSIM=$(python -c "import json;print(len(json.load(open('$CONFIG'))['sims']))")
ARRAY=${1:-0-$((NSIM*NCHUNK-1))}
TIME=${TIME:-16:00:00}
CPUS=${CPUS:-10}
MEM=${MEM:-20G}
LOGS=$(python -c "import json;print(json.load(open('$CONFIG'))['output_dir'])")/logs_co
mkdir -p $LOGS
sbatch -J co_p79d -p shared -c $CPUS --mem=$MEM -t $TIME -A phy240036 --array=$ARRAY -o $LOGS/co_%a.out \
       --wrap "source ~/.bashrc; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$CPUS; \
               export TMPDIR=/tmp/co_\${SLURM_JOB_ID}_\${SLURM_ARRAY_TASK_ID}; mkdir -p \$TMPDIR; \
               python extract_sim_co.py extract --config $CONFIG --rt $RT \
                      --index \$((SLURM_ARRAY_TASK_ID / $NCHUNK)) --chunk \$((SLURM_ARRAY_TASK_ID % $NCHUNK)) \
                      --n_chunks $NCHUNK $EXTRA; rm -rf \$TMPDIR"
