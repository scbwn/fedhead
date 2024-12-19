This repository implements FedHEAD and FedHEAD+ in Python 3.8.

# Description of Files

## fedlib.py - Implements FedHEAD, FedHEAD+ and their comparison benchmarks for cv tasks.

## fedlib_new.py - Implements FedHEAD, FedHEAD+ and their comparison benchmarks for nlp and ntc tasks.

## cv_tasks.py - Contains driver code for running experiments on cv tasks.

## nlp_tasks.py - Contains driver code for running experiments on nlp tasks.

## ntc_tasks.py - Contains driver code for running experiments on ntc tasks.

## n_clients.py - Contains driver code for running experiments varying number of clients.

## n_sectors.py - Contains driver code for running experiments varying number of sectors.


# Submit jobs using SLURM

Submit job array using slurm command "python -u SCRIPT_NAME $SLURM_ARRAY_TASK_ID" where $SLURM_ARRAY_TASK_ID is the job id mentioned in the corresponding script.
