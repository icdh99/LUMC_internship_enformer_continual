#!/bin/bash

# supp_14a_txt=/exports/humgen/idenhond/data/DAR/AC_level/supp_14a.txt

# supp_14a_bed_unsorted=/exports/humgen/idenhond/data/DAR/AC_level/supp_14a_unsorted.bed

# supp_14a_bed_sorted=/exports/humgen/idenhond/data/DAR/AC_level/supp_14a_sorted.bed

# awk -F'[:-]' ' {print $1 "\t" $2 "\t" $3}' $supp_14a_txt > $supp_14a_bed_unsorted

awk -F'[:-]' '{ print $1 "\t" $2 "\t" $3 >> $NF".bed" }' /exports/humgen/idenhond/data/DAR/AC_level/supplementary_table_14A.csv



