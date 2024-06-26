
ENV_NAME='motip'
conda env remove -n $ENV_NAME -y
conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME

# PyTorch:
module load cuda/11.8
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y		# CUDA version=12.1 is also OK

# Other dependencies:
conda install matplotlib pyyaml scipy tqdm tensorboard seaborn scikit-learn pandas -y
pip install opencv-python einops wandb pycocotools timm

# # Compile the Deformable Attention:
# cd models/ops/
# sh make.sh

# # After compiled, you can use following script to test it:
# python test.py		# [Optional]