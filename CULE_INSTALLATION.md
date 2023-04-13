Make sure to set up the pythonpath every time you start a new bash environment or you won't be able to import CULE. (Might want to just add it to bashrc)

```
module load pytorch cuda/10.0 gcc/7.4.0
pip3 install psutil pytz tqdm atari_py gym
mkdir ~/CULE_INSTALL
git clone --recursive https://github.com/NVlabs/cule
cd cule
python3 setup.py install --fastbuild --prefix ~/CULE_INSTALL
export PYTHONPATH=$PYTHONPATH:~/CULE_INSTALL/lib/python3.9/site-packages/torchcule-0.1.0-py3.9-linux-x86_64.egg/
```

Then you need to get import Atari ROM files
```
git clone https://github.com/agaction/cse6230-spring23-final-proj-ddppo.git
cd ~/cse6230-spring23-final-proj-ddppo/roms
python3 -m atari_py.import_roms .
```
