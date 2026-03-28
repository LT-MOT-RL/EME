# EME
The details will be further refined.

#  Install the environment


CUDA 11.3
Partial paramount site-packages requirements are listed below:
- `python == 3.9.7` 
- `pytorch == 1.11.0`
- `torchvision == 0.12.0`
- `matplotlib == 3.5.1`
- `numpy == 1.21.2`
- `pandas == 1.4.1`
- `pyyaml == 6.0`
- `scipy == 1.7.3`
- `scikit-learn == 1.0.2`
- `tqdm == 4.63.0`
- `yaml == 0.2.5`
- `opencv-python == 4.5.5.64`

#  OTETrack 

Thanks to the contributors of [OTETrack](https://github.com/OrigamiSL/OTETrack).  
Please place the file `OTETrack_all.pth.tar` into the `./OTETrack/test_checkpoint/` directory.
For more details, please refer to [OTETrack](https://github.com/OrigamiSL/OTETrack).


1. Weight sources (OTETrack_256_full):

You can download the model weights from [Google Drive](https://drive.google.com/file/d/1-9CceF4HwsudLi9pt5ylDEhYtrgGDhsz/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1lJz4RlgCE8XW7lV3sXbcBw?pwd=25ur) (extracted code: 25ur).

Put the model weights you download in `./test_checkpoint.` The file tree shall look like this:
```
   ${PROJECT_ROOT}
    |-- test_checkpoint
    |   |-- test_checkpoint
    |   |   |-- OTETrack_all.pth.tar
```




#  Unicorn

Thanks to the contributors of [Unicorn](https://github.com/MasterBin-IIAU/Unicorn). 
Deploying Unicorn can be difficult. Please see [Unicorn](https://github.com/MasterBin-IIAU/Unicorn) for guidance.

You can download the model weights from [Unicorn.model_zoo.md](https://github.com/MasterBin-IIAU/Unicorn/blob/master/assets/model_zoo.md).




1. Install Deformable Attention

cd unicorn/models/ops
bash make.sh
cd ../../..


2. Install mmcv, mmdet, bdd100k

cd external_2/qdtrack
wget -c https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/mmcv_full-1.4.6-cp37-cp37m-manylinux1_x86_64.whl # This should change according to cuda version and pytorch version
pip3 install --user mmcv_full-1.4.6-cp37-cp37m-manylinux1_x86_64.whl
pip3 install --user mmdet
git clone https://github.com/bdd100k/bdd100k.git
cd bdd100k
python3 setup.py develop --user
pip3 uninstall -y scalabel
pip3 install --user git+https://github.com/scalabel/scalabel.git
cd ../../..

The downloaded checkpoints should be organized in the following structure
```
${UNICORN_ROOT}
 -- Unicorn_outputs
     -- unicorn_det_convnext_large_800x1280
         -- best_ckpt.pth
     -- unicorn_det_convnext_tiny_800x1280
         -- best_ckpt.pth
     -- unicorn_det_r50_800x1280
         -- best_ckpt.pth
     -- unicorn_track_large_mask
         -- latest_ckpt.pth
     -- unicorn_track_tiny_mask
         -- latest_ckpt.pth
     -- unicorn_track_r50_mask
         -- latest_ckpt.pth
     ...
```




#  Test
```
python tools/test_lasot.py
```

# Others (Train) 

Coming soon~~