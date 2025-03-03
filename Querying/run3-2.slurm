#!/bin/bash
#SBATCH --job-name=Mislam_thesis_job    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mislam.mail@uni-mannheim.de     # Where to send mail	
#SBATCH --error=output/job.%J.err
#SBATCH --output=output/job.%J.out
#SBATCH --partition=single # or some other partition that is idle (can be seen by typing sinfo_t_idle)
#SBATCH --time=48:00:00 # the estimated time
#SBATCH --mem=126G # the estimated memory
#SBATCH --nodes=1 
#SBATCH --ntasks=1 # Usually one node and one task were enough


echo "Running plot script on a single CPU core"
# python /pfs/work7/workspace/scratch/ma_mislam-mislam_twitterlm_thesis/Code/masters_thesis_code/Fine_tuning_bert_new.py
python test_withSimOld_epoch2.py

date