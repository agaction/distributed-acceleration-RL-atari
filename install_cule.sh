#!/bin/bash

# Install Dependencies
pip install --user cython psutil pytz tqdm atari_py numpy==1.17.0 opencv-python torch==1.2.0 gym==0.14.0 pynvml

# Clone CULE
git clone --recursive https://github.com/NVlabs/cule
cd cule

# Modify CUDA architecture support
sed -i -e "s/torch.cuda.get_arch_list() if torch.cuda.device_count() > 0 else//" setup.py

# Build and Install CULE, It will throw a lot of warnings and  seem like it has frozen for up to 20 minutes, just let it run
pip install -v -e . --user --install-option='--fastbuild'

# Import the Atari ROM File
cd ../roms
python -m atari_py.import_roms .
