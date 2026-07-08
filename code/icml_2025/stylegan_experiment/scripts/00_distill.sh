cd modules/farel_extra
python predict_posteriors.py age_resnet50.jit distillation_data

python src/train_regression_oracle.py --image_dir distillation_data --npz_path distillation_data/predictions.npz --ckpt_path modules/farel_extra/age_resnet50.pth --epochs 100
