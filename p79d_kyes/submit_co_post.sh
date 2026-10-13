#!/bin/sh
#-------------------------------------------------------------------------------
# Run 2 post-processing, one job: merge the noise-free dataset, then for each
# observe setup in radmc_co_v2.json re-observe the stored cubes and merge.
#
#   sh submit_co_post.sh                          # merge + all setups
#   SETUPS="mwisp" SKIP_MERGE=1 sh submit_co_post.sh
#-------------------------------------------------------------------------------
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
CONFIG=sims_mach_grid.json
RT=radmc_co_v2.json
SETUPS=${SETUPS:-"fcrao500 mwisp nro_orion"}
LOGS=$(python -c "import json;print(json.load(open('$CONFIG'))['output_dir'])")/logs_co
CMD="source ~/.bashrc; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1; set -e;"
[ -z "$SKIP_MERGE" ] && CMD="$CMD python extract_sim_co.py merge --config $CONFIG --rt $RT;"
for S in $SETUPS; do
  CMD="$CMD python extract_sim_co.py reobserve --config $CONFIG --rt $RT --obs $S; \
            python extract_sim_co.py merge --config $CONFIG --rt $RT --obs $S;"
done
sbatch -J co_post -p shared -c 32 --mem=60G -t ${TIME:-03:00:00} -A phy240036 -o $LOGS/co_post.out --wrap "$CMD"
