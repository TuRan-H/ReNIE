# python==3.9.19
source ~/.anaconda3/bin/activate
conda activate GoLLIE


# pytorch
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# numpy
pip install numpy==1.19.5

# faiss
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y

# pandas
conda install pandas==1.4.4 -y

# simcse
pip install simcse==0.4

# transformers
pip install transformers==4.33.2

# peft
pip install peft==0.10.0

# bitsandbytes
pip install bitsandbytes==0.43.1

# flash-attn
pip install flash-attn --no-build-isolation

# spacy
pip install spacy
python -m spacy download en_core_web_sm

# else package
pip install black
pip install Jinja2
pip install tqdm
pip install rich
pip install psutil
pip install datasets==2.18.0
pip install ruff
pip install wandb
pip install fschat
pip install libcst
pip install astor
pip install nltk==3.8.1
pip install pyarrow==15.0.2
pip install accelerate==0.29.2