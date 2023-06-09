#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --time=01:00:00
#SBATCH --mem=400G

python /exports/humgen/idenhond/projects/enformer/weigths/plot_weights_figure1.py
