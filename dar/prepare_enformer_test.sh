#!/bin/bash

enformer_bed=/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac/sequences.bed
enformer_test_bed=/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_test.bed
enformer_valid_bed=/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_valid.bed
enformer_train_bed=/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_train.bed

awk -F'\t' '$NF == "test" { print $1 "\t" ($2 + 8192) "\t" ($3 - 8192) }' $enformer_bed > $enformer_test_bed
awk -F'\t' '$NF == "valid" { print $1 "\t" ($2 + 8192) "\t" ($3 - 8192) }' $enformer_bed > $enformer_valid_bed
awk -F'\t' '$NF == "train" { print $1 "\t" ($2 + 8192) "\t" ($3 - 8192) }' $enformer_bed > $enformer_train_bed



wc -l $enformer_test_bed
wc -l $enformer_valid_bed
wc -l $enformer_train_bed