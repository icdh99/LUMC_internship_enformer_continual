#!/bin/bash


supp_14a_txt=/exports/humgen/idenhond/data/DAR/supp_14a.txt
supp_14a_bed_unsorted=/exports/humgen/idenhond/data/DAR/supp_14a_unsorted.bed
supp_14a_bed_sorted=/exports/humgen/idenhond/data/DAR/supp_14a_sorted.bed

awk -F'[:-]' ' {print $1 "\t" $2 "\t" $3}' $supp_14a_txt > $supp_14a_bed_unsorted
sort-bed $supp_14a_bed_unsorted > $supp_14a_bed_sorted

enformer_bed=/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac/sequences.bed
enformer_test_bed_unsorted=/exports/humgen/idenhond/data/DAR/enformer_test_unsorted.bed
enformer_test_bed_sorted=/exports/humgen/idenhond/data/DAR/enformer_test_sorted.bed
awk -F'\t' '$NF == "test" { print $1 "\t" $2 "\t" $3 }' $enformer_bed > $enformer_test_bed_unsorted

sort-bed $enformer_test_bed_unsorted > $enformer_test_bed_sorted

# bedops --ec --everything $supp_14a_bed_sorted | head
# bedops --ec --everything $enformer_test_bed_sorted

rm $enformer_test_bed_unsorted
rm $supp_14a_bed_unsorted

output_intersect=/exports/humgen/idenhond/data/DAR/intersect_14a_enformertest.bed

bedtools intersect -a $enformer_test_bed_sorted -b $supp_14a_bed_sorted > $output_intersect
