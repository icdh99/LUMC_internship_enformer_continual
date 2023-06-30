Analysis of differentially accessible regions

This analysis is based on the DARs defined in Bakken et al. 
For the subclass DARs: we use Supplementary table 14E from the Bakken paper (/exports/humgen/idenhond/data/DAR/Subclass_level/Subclass_neuronal_14E.bed)
Taking the intersect with the Enformer test set: /exports/humgen/idenhond/data/DAR/Subclass_level/intersect_14E_test_wb_sorted.bed

For the AC-level DARs we use Supplementary table 14A from the Bakken paper (/exports/humgen/idenhond/data/DAR/AC_level/Cluster_14A.bed)
Taking the intersect with the Enformer test set: /exports/humgen/idenhond/data/DAR/AC_level/intersect_14A_test_withsubclasses_sorted.bed

Enformer test/validation/train sequences are in this folder: /exports/humgen/idenhond/data/DAR/Enformer_test and generated with /exports/humgen/idenhond/projects/dar/prepare_enformer_test.sh

The prepare_bedfiles folder contain scripts to make and analyse the intersect between the DARs and the Enformer test sequences for both the subclass and AC-level DARs. 

The correlation_per_DAR folders contain the code to make the final heatmaps and to analyse the correlation per DAR for both the subclass and AC-level DARs

AC-level DARs:
/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/correlation_per_DAR_aclevel.py and correlation_per_DAR_aclevel_allenformerseqs.py
- stores the predicted and observed values for each DAR in Predictions_DAR_aclevel.csv (all Enformer), Predictions_test_DAR_aclevel.csv (Enformer test), and Targets_DAR_aclevel.csv (all Enformer), Targets_test_DAR_aclevel.csv (Enformer test). 

/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/plot_correlation_test_DAR_aclevel_allenformerseqs.py and plot_correlation_DAR_aclevel_allenformerseqs.py
- plots the heatmaps with the observed and predicted chromatin accessibility values per DAR/cluster, the histogram with correlation per DAR, and the example DARs.

/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/compare_pred_target_cluster.py
- stores for every DAR the highest observed and predicted class in /exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Clusters_pred_target_ac_test.txt and Clusters_pred_target_ac.txt

/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/plot_pred_target_clusters.py
- plots the confusion matrix for the 'enriched' observed and predicted clusters 

The subclass DAR analysis is similar in the Subclass folder


