cd ../train

# Here we use Car dataset as an example
# Adjust the hyperparameters accordingly for other datasets

# Standard training
for split in 0 1 2; do
    python train_baseline.py \
        --gpu 0 1 \
        --dataset car \
        --split ${split} \
        --epochs 90 \
        --batch-size 32
done

# Rotation training
for split in 0 1 2; do
    python train_rot.py \
        --gpu 0 1 \
        --dataset car \
        --split ${split} \
        --epochs 90 \
        --batch-size 32
done

# OE training
for split in 0 1 2; do
    python train_oe.py \
        --gpu 0 1 \
        --dataset car \
        --split ${split} \
        --epochs 10 \
        --batch-size 32 \
        --beta 1.0
done

# EnergyOE training
for split in 0 1 2; do
    python train_oe.py \
        --gpu 0 1 \
        --dataset car \
        --split ${split} \
        --epochs 10 \
        --batch-size 32 \
        --beta 0.1 \
        --energy \
        --m_in -13 \
        --m_out -6
done

# OE-M training
for split in 0 1 2; do
    python train_oem.py \
        --gpu 0 1 \
        --dataset car \
        --split ${split} \
        --epochs 10 \
        --batch-size 32 \
        --beta 1.0
done

# MixOE training
for split in 0 1 2; do
    python train_mixoe.py \
        --gpu 0 1 \
        --dataset car \
        --split ${split} \
        --epochs 10 \
        --batch-size 32 \
        --mix-op mixup \
        --alpha 1.0 \
        --beta 5.0

    python train_mixoe.py \
        --gpu 0 1 \
        --dataset car \
        --split ${split} \
        --epochs 10 \
        --batch-size 32 \
        --mix-op cutmix \
        --alpha 2.0 \
        --beta 5.0
done