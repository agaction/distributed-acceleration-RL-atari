```
module load cuda/10.0 gcc/7.4.0 anaconda3/2019.07
pip install cython psutil pytz tqdm atari_py gym opencv-python torch==1.2.0
mkdir ~/CULE_INSTALL
git clone --recursive https://github.com/NVlabs/cule
cd cule
```

Open the setup.py file in the cule repo and change line 12 to be just `gpus =  ['70']`

```
pip install -v -e . --user --install-option='--fastbuild'
```

Then you need to import Atari ROM files (I think we might need to go find a different set of roms, the ones I added aren't the best option)
```
git clone https://github.com/agaction/cse6230-spring23-final-proj-ddppo.git
cd ~/cse6230-spring23-final-proj-ddppo/roms
python -m atari_py.import_roms .
```
