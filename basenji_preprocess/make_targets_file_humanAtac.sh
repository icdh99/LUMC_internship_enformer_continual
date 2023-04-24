#!/bin/bash

index=0 # Initialize the counter
output_file="/exports/humgen/idenhond/data/basenji_preprocess/targets_human_atac.txt" # Set the name of the output file

# Generate the header line
header="index\tidentifier\tfile\tclip\tscale\tsum_stat\tdescription"

# Write the header line to the output file
echo -e "$header" > "$output_file"

# Loop over each file in the directory
for file in /exports/humgen/idenhond/data/human_Mop_ATAC/bw_files/*
do
    identifier=$(basename "$file" .bw) # Get the filename without path
    
    file_path=$(realpath "$file") # Get the full path to the file
    clip=32
    scale=2
    sum_stat="mean"
    description="$identifier"

    # Generate the line for the current file
    line="$index\t$identifier\t$file_path\t$clip\t$scale\t$sum_stat\t$description"

    # Append the line to the output file
    echo -e "$line" >> "$output_file"

    # Increment the counter
    ((index++))
done