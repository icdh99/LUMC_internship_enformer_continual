#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --time=6:00:00
#SBATCH --mem=10G

count=0

while read p || [[ -n $p ]]
do
    file=$p
    filename=$(echo $file | cut -d'/' -f5 | cut -d'?' -f1)
    echo $filename
    wget --no-verbose -O /exports/humgen/idenhond/data/human_Mop_ATAC/bw_files/$filename $file 
    count=$((count+1))
done < /exports/humgen/idenhond/data/human_Mop_ATAC/filenames/human_Mop_ATAC.txt

echo "Total number of lines: $count"