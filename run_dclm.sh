#!/bin/bash
#SBATCH --job-name=8b_8b-reduced
#SBATCH --output=/n/netscratch/sham_lab/Everyone/cbrownpinilla/CF_DCLM/dclm_color_filter_olmo/logs/dclm_runs/%x_%A_%a.log
#SBATCH -p kempner_h100
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4     
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=250GB		
#SBATCH --constraint=h100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=camilobrownpinilla@college.harvard.edu
#SBATCH --array=1-1

# Custom environment
source ~/.bashrc
module load python/3.10.13-fasrc01
conda deactivate
conda activate color-filter
cd DCLM/

# Set to path to output log
export SLURM_OUTPUT_FILE="/n/netscratch/sham_lab/Everyone/cbrownpinilla/CF_DCLM/dclm_color_filter_olmo/logs/dclm_runs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

python ../dclm.py \
    --selected_dir /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/selected/core-train-tasks/8b-reduced/8b \
    --dclm_scale 411m_1x\
    --evaluation heavy\
    --multiple_data_passes
