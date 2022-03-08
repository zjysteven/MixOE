cd ../eval

# MSP, ODIN, Energy
python eval.py \
    --gpu 0 1 \
    --model-file checkpoints/car/split_0/rn_baseline_epochs_90_bs_32/seed_0.pth \
    #--ood odin --temp 1000   # uncomment this line if using ODIN
    #--ood energy --temp 1    # uncomment this line if using Energy
    #-s                       # uncomment this line if you want to save results to file
    
# OE
python eval.py \
    --gpu 0 1 \
    --model-file checkpoints/car/split_0/rn_oe_WebVision_beta=1.0_epochs_10_bs_32/seed_0.pth \

# EnergyOE
python eval.py \
    --gpu 0 1 \
    --model-file checkpoints/car/split_0/rn_oe_WebVision_energy_min=-13_mout=-6_beta=0.1_epochs=10_bs=32/seed_0.pth \
    --ood energy

# OE-M
python eval.py \
    --gpu 0 1 \
    --model-file checkpoints/car/split_0/rn_oem_WebVision_beta=1.0_epochs_10_bs_32/seed_0.pth

# MixOE
python eval.py \
    --gpu 0 1 \
    --model-file checkpoints/car/split_0/rn_mixup_WebVision_alpha=1.0_beta=5.0_epochs_10_bs_32/seed_0.pth
    #--model-file checkpoints/car/split_0/rn_cutmix_WebVision_alpha=2.0_beta=5.0_epochs_10_bs_32/seed_0.pth