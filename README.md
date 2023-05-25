# 🌍 Qartezator: Yet another aerial image-to-map translator

Qartezator is your translator between aerial images and maps.

![Qartezator teaser](https://github.com/AndranikSargsyan/qartezator/blob/master/assets/teaser.gif)

# Environment setup

Clone the repo: `git clone https://github.com/AndranikSargsyan/qartezator.git`

Set up virtualenv:
```bash
cd qartezator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```

If you need torch+cuda, you can use the following command
```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

# Demo
```bash
streamlit run demo.py
```

# Single Inference
```bash
python -m qartezator.inference -m PATH-TO-MODEL -i PATH-TO-IMAGE -o OUTPUT-PATH
```

# Training   

```bash
python -m qartezator.train --config-path ./qartezator/configs/qartezator-fourier.yaml
```