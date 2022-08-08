#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=genoCP1
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=80000m 
#SBATCH --time=24:00:00
#SBATCH --account=welchjd1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# The application(s) to execute along with its input arguments and options:

module list
python --version
printenv
python genotypeVAE_ClusterProperty_FineTune_copy_copy_removed.py
