In order to run the experiments, please clone the StyleGAN3 repository to modules/stylegan3. Afterwards, install the stylegan3 conda environment.

Afterwards, the experiments can be run by running:
```
01_generate.sh
02_label.sh
03_train_resnet.sh
03_train_farl.sh
```

Due to size constraints of the ICML conference supplementary, we provide the already distilled resnet50 model in the ZIP, instead of providing the original model before distillation.

