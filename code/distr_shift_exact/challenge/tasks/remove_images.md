# Plan 
make the source images inaccessible for the challenge contestans. 

# Context
The current version of the challenge gives the students the development images, the test images as well as the calibrated model. The studens also get predictions of the base predictots. They can however generate the predictions themself. 

The source of the TissueMNIST dataset has to be revealed due to lincence. It is thus easy for the students to recover true labels of the data. To prevent this, the students will get only the predcitions (i.e. csv file with the posterior for development and test images). They will not any images which makes it difficult for them to recover the true label.

# Task

Change the code base and relevat files appropriately. The `out/kaggle/`
will not contain the images. The starter kit will not contain the trained model. 