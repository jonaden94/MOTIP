
ENV_NAME='motip'
conda env remove -n $ENV_NAME -y
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# PyTorch:
module load cuda/11.8 # necessary to make cluster cuda version compatible with the installed pytorch version
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y		# CUDA version=12.1 is also OK
conda install mkl=2024.0.0 -y # needs to downgraded because of some bug in other version

# Other dependencies:
conda install matplotlib pyyaml scipy tqdm tensorboard seaborn scikit-learn pandas -y
pip install opencv-python einops wandb pycocotools timm jupyter

# # Compile the Deformable Attention:
# cd models/ops/
# sh make.sh

# # After compiled, you can use following script to test it:
# python test.py		# [Optional]