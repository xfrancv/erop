# Plan

Create scripts which download and analyze the datasets in 'tasks/dataset_proposal.md' .

# Tasks

There will be a script which:
1. creates folder 'data/'
2. attempts to download the datasets specified in:
'tasks/dataset_proposal.md'
3. for each dataset it creates a script which analyzes the data. The output of the script will be a self-contained HTML. It will contain: 
* the number of examples ; if the data are split in to training, test, or validation subsets it reports the number of examples in each subset
* dimensinality
* the number of labels (if it is the classification problem)
* it visualizes an example of the input and output; e.g. if it is image database, it shows a few examples of images in each class; 
