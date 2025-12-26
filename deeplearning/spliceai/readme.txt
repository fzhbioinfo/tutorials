# 第一个位置获取不到
awk '{print $3"\t"$5-5001"\t"$6+5000}' canonical_dataset.txt > canonical_dataset.slop5000.bed
bedtools getfasta -bed canonical_dataset.slop5000.bed -fi /jdfstj4/B2C_RD_R1/fangzhonghai/pipeline/db/GRCh37/ref/ref.fa -fo canonical_dataset.slop5000.sequences.txt -tab
paste canonical_dataset.txt canonical_dataset.slop5000.sequences.txt|cut -f1-8,10 > canonical_dataset.add_slop5000sequences.txt

