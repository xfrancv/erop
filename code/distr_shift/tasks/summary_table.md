# Plan

Create a script which combines text files containing tables with metrics to summary table.

# Context

The 'run_real_reject_option_exp.py' when run in sweep mode reports results to a text file. An examples of a report file is here:
runs/bloodmnist/20260727_114317/real_reject_option_sweep_report.txt

The file contains several metrics which measure performance of the
benchmarked predictors for given adaptation set size $n_test$.
The main metrics is the Area Under Regret-coverage curve, 
reported in table titled 'AuRC (regret)  [x1000]' and 
'win% AuRC (regret)'.

The goal is to take subset of lines from these values and organized them 
to a single summary table.

# Task
Create a script 'summary_table.py' which has the following arguments:
1. --reports file1.txt file2.txt ... fileN.txt   defining a set of report files.
2. --sizes n1 n2 .. nK   defines numbers of test set size, each identifying a single line
3. --output file_name.txt   output file to store result summary.

From each report file, the script takes from table 'AuRC (regret)  [x1000]':
- AuRC for the Epistemic reject-option predictor (column 'Bayesian, epistemic un')
- AuRC for the Bayesian reject-option predicotr (column ' Bayesian, total uncert')
and from table 'win% AuRC (regret)' it takes the second column representing the 
percentage of wins.
Only the lines defined by --n-size are taken from the tables.

The script outputs a summary table with a format described in the following example.
Invoking 
```
python summary_table.py --reports runs/bloodmnist/20260727_114317/real_reject_option_sweep_report.txt \
runs/cifar10/20260727_114317/real_reject_option_sweep_report.txt \
--size 1 10 --output table.txt
```
will output 'table.txt' with the following content:

```
           & \multicolumn{3}{c}{n=1} & \multicolumn{3}{c}{n=10} \\
dataset    & Epist  & Bayes   & Win [\%] & Epist  & Bayes   & Win[\%] \\
bloodmnist & 0.1714 & 0.33331 & 70.0  & 0.1188 & 0.2254 & 70.0  \\
cifar10    & 1.2287 & 2.3679  & 90.0  & 0.6465 & 1.3678 & 80.0  \\
```
where the dataset names like 'bloodmnist' are derived from the line 
'command : run_real_reject_option_exp.py ...' of the report file ;
the column 'Epist' contains the AuRC for the episteminc reject option predictor;
the column 'Bayes' contains the AuRC for the Baysian reject option predictor;
the column 'Win [\%]' contains the 100 minus percentage of the wins taken from the 'win% AuRC (regret)'
table. 


