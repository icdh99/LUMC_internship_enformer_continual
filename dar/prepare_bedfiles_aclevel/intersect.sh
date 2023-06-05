#!/bin/bash

enformer_test_bed=/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_test.bed
supp_14e_bed=/exports/humgen/idenhond/data/DAR/AC_level/Cluster_14A.bed
output_intersect=/exports/humgen/idenhond/data/DAR/AC_level/intersect_14A_test.bed
output_intersect_wb=/exports/humgen/idenhond/data/DAR/AC_level/intersect_14A_test_withsubclasses.bed
output_intersect_wb_sorted=/exports/humgen/idenhond/data/DAR/AC_level/intersect_14A_test_withsubclasses_sorted.bed

bedtools intersect -a $enformer_test_bed -b $supp_14e_bed > $output_intersect

bedtools intersect -wb -a $enformer_test_bed -b $supp_14e_bed > $output_intersect_wb

wc -l $output_intersect
wc -l $output_intersect_wb

awk '{if (NR==FNR) a[$1$2$3]; else if (!($1$2$3 in a)) print}' $output_intersect $output_intersect_wb

sort -t$'\t' -k 8 $output_intersect_wb > $output_intersect_wb_sorted