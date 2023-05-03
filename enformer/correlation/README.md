This folder contains scripts to reproduce the correlation results from the Enformer paper

Code is adjusted from ```` /exports/humgen/idenhond/enformer_dev/enformer-pytorch/evaluate_enformer_pytorch_correlation.ipynb ```` to work with local file structure

16/2: evaluate_correlation.py
- calculate correlation coefficient with sequence and target from tensor flow records
- model output is generated on the fly 

17/2: evaluate_correlation.py
- calculate correlation coefficient per track (5313 tracks)

20/2: evaluate_correlation_own_output.py
- calculate correlation coefficient with target from tensor flow records and model output from /exports/humgen/idenhond/data/Enformer_test/Enformer_test_output
- untrained model

21/2: evaluate_correlation_random_seqs
- calculate correlation coefficient with random sequences and targets from tfr files

22/2: evaluate_correlation_own_output_pretrainedmodel.py
- newmodel --> pretrainedmodel in names
- calculate correlation coefficient with output generated with pretrained model and targets from tfr files
- good results
- analyse correlation coefficient in analyse_tracks_correlation_perassaytype_ownoutputpretrainedmodel.py

TODO:
- merge analyse_tracks_correlation_own_output_pretrainedmodel.py (plots) and analyse_tracks_correlation_perassaytype_ownoutputpretrainedmodel.py (corr per assay type)
- plot corr per track per assay type (split from paper) and add corr score in figure
- calculate correlation coefficient per position (128 bp bin) per track

13/2: evaluate_correlation_dnn_head.py
- use model (dnn head trained on train + valid sequences from enformer) to get enformer output for test sequences 
- generate model output on the fly
- calculate correlation between model output + tensor flow record output
- save correlation scores per track to csv file
- analyse correlation scores

27/3: evaluate_correlation_dnn_head_newtracks.py
- evaluate model trained on new tracks (19) on 27/3

30/3: evaluate_correlation_dnn_head_alltracks.py
- evaluate model trained on all tracks on 30/3

31/3: evaluate_correlation_dnase.py
- evaluate model trained on DNASE tracks only

25/4: evaluate_correlation_humanatac.py
- use model trained on human ATAC-seq tracks to get predicted output for test, train and validation
- calculate correlation between model output + tfr target

25/4: evaluate_correlation_newtracks_2404.py
- use model trained on 27 new tracks to get predicted output for test, train and validation
- calculate correlation between model output + tfr target

28/4: evaluate_correlation_alltracks_2404.py
- evaluate model trained on all tracks 2404

1/5: evaluate_correlation_humanatac_normalized.py
- evaluate model trained on normalized human atac tracks (did not work)

2/5: evaluate_correlation_newtracks_newmodel_0205.py
- evaluate model with more layers trained on 27 new tracks