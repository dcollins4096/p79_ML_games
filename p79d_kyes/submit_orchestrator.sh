#!/bin/sh
#-------------------------------------------------------------------------------
# Run orchestrate_net9010.py as a small long job (1 core, 2 GB, shared), so the
# loop survives logouts. Restartable: all progress is in the state file.
#
#   python orchestrate_net9010.py plan --plan run2 --out <state.json>   # once
#   sh submit_orchestrator.sh <state.json>
#   python orchestrate_net9010.py status --state <state.json>
#-------------------------------------------------------------------------------
STATE=${1:?usage: sh submit_orchestrator.sh <state.json>}
cd /home/x-nbisht1/scripts/p79_nikhil/p79_ML_games/p79d_kyes
sbatch -J orch_$(basename $STATE .json) -p shared -c 1 --mem=2G -t ${TIME:-12:00:00} -A phy240036 \
       --open-mode=append -o /home/x-nbisht1/projects/p79d_dataset/models/logs/$(basename $STATE .json).log \
       --wrap "source ~/.bashrc; export PYTHONUNBUFFERED=1; python orchestrate_net9010.py run --state $STATE --max_hours 11.5"
