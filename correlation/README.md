This folder contains scripts to reproduce the correlation results from the Enformer paper

Code is adjusted from ```` /exports/humgen/idenhond/enformer_dev/enformer-pytorch/evaluate_enformer_pytorch_correlation.ipynb ```` to work with local file structure

16/2: evaluate_correlation.py
- calculate correlation coefficient with sequence and target from tensor flow records
- model output is generated on the fly 

17/2: evaluate_correlation.py
- calculate correlation coefficient per track (5313 tracks)

17/2: evaluate_correlation_own_output.py -> nog niet aan toegekomen
- calculate correlation coefficient with target from tensor flow records and model output from /exports/humgen/idenhond/data/Enformer_test/Enformer_test_output
