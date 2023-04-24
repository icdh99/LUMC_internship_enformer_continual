
while IFS=$'\t' read -r chr start end _; do
  start=$((start + 8192))
  end=$((end - 8192))
  printf "%s:%d-%d\n" "$chr" "$start" "$end"
done < /exports/humgen/idenhond/data/Basenji/sequences.bed > /exports/humgen/idenhond/data/Basenji/sequences_cropped.txt