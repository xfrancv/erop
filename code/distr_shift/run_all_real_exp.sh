sbatch -J erop-blood ./run_real_exp.sh bloodmnist
sbatch -J erop-blood ./run_real_exp.sh bloodmnist noadapt
sbatch -J erop-blood ./run_real_exp.sh bloodmnist beta

sbatch -J erop-cif10 ./run_real_exp.sh cifar10
sbatch -J erop-cif10 ./run_real_exp.sh cifar10 noadapt
sbatch -J erop-cif10 ./run_real_exp.sh cifar10 beta

sbatch -J erop-cif100 ./run_real_exp.sh cifar100
sbatch -J erop-cif100 ./run_real_exp.sh cifar100 noadapt
sbatch -J erop-cif100 ./run_real_exp.sh cifar100 beta

sbatch -J erop-derma ./run_real_exp.sh dermamnist
sbatch -J erop-derma ./run_real_exp.sh dermamnist noadapt
sbatch -J erop-derma ./run_real_exp.sh dermamnist beta

sbatch -J erop-fashion ./run_real_exp.sh fashion_mnist
sbatch -J erop-fashion ./run_real_exp.sh fashion_mnist noadapt
sbatch -J erop-fashion ./run_real_exp.sh fashion_mnist beta

sbatch -J erop-tissue ./run_real_exp.sh tissuemnist
sbatch -J erop-tissue ./run_real_exp.sh tissuemnist noadapt
sbatch -J erop-tissue ./run_real_exp.sh tissuemnist beta

sbatch -J erop-organa ./run_real_exp.sh organamnist
sbatch -J erop-organa ./run_real_exp.sh organamnist noadapt
sbatch -J erop-organa ./run_real_exp.sh organamnist beta

sbatch -J erop-organa ./run_real_exp.sh organsmnist
sbatch -J erop-organa ./run_real_exp.sh organsmnist noadapt
sbatch -J erop-organa ./run_real_exp.sh organsmnist beta
