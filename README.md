# EME
The dataset generation and training code will be added.

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

# OTETrack

Thanks to the contributors of [OTETrack](https://github.com/OrigamiSL/OTETrack).  
For more details, please refer to [OTETrack](https://github.com/OrigamiSL/OTETrack).

Download the `OTETrack_256_full` weights from [Google Drive](https://drive.google.com/file/d/1-9CceF4HwsudLi9pt5ylDEhYtrgGDhsz/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1lJz4RlgCE8XW7lV3sXbcBw?pwd=25ur) (extraction code: `25ur`). Rename the file to `OTETrack_all.pth.tar` if necessary and place it as follows:

```
${PROJECT_ROOT}/
|-- OTETrack/
    |-- test_checkpoint/
        |-- OTETrack_all.pth.tar
```

# Unicorn

Thanks to the contributors of [Unicorn](https://github.com/MasterBin-IIAU/Unicorn). 
Deploying Unicorn can be difficult. Please see [Unicorn](https://github.com/MasterBin-IIAU/Unicorn) for guidance.

Install Deformable Attention from the EME project root:

```bash
cd Unicorn/unicorn/models/ops
bash make.sh
cd ../../../..
```

Install the remaining Unicorn dependencies. The MMCV build must match the Python, PyTorch, and CUDA versions of your environment; do not use a wheel built for a different Python or PyTorch version.

```bash
cd Unicorn/external_2/qdtrack
# Install a compatible mmcv-full build before mmdet.
pip install mmdet
git clone https://github.com/bdd100k/bdd100k.git
cd bdd100k
python setup.py develop --user
pip uninstall -y scalabel
pip install --user git+https://github.com/scalabel/scalabel.git
cd ../../../..
```

`tools/test_lasot.py` uses the `unicorn_track_tiny_sot_only` experiment. Download the corresponding checkpoint following the [Unicorn model zoo](https://github.com/MasterBin-IIAU/Unicorn/blob/master/assets/model_zoo.md) and place it at this exact path:

```
${PROJECT_ROOT}/
|-- Unicorn/
    |-- Unicorn_outputs/
        |-- unicorn_track_tiny_sot_only/
            |-- latest_ckpt.pth
```

# LaSOT

Place the LaSOT dataset under `datasets/LaSOT`. The expected layout is:

```
${PROJECT_ROOT}/
|-- datasets/
    |-- LaSOT/
        |-- airplane/
        |   |-- airplane-1/
        |       |-- groundtruth.txt
        |       |-- img/
        |-- ...
```

# Test

Run the test from the EME project root:

```bash
python tools/test_lasot.py
```

# Others (Train)

Coming soon.
