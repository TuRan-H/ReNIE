# Environment Setup
```bash
conda create -n GoLLIE python==3.9.*
bash requirements.sh
conda activate GoLLIE
```

# Running the Code
## Generating Data
```
bash scripts/generate_data.sh
```

Content of the bash file
```bash
#!/bin/bash

source ~/.anaconda3/bin/activate
conda activate GoLLIE

if [ -f .env ]; then
	export $(cat .env | grep -v '^[#;]' | xargs)
fi

CONFIG_DIR="configs/data_configs"
TOP_K=1

python -m src.generate_data \
	--configs \
		${CONFIG_DIR}/ace_config.json \
		${CONFIG_DIR}/bc5cdr_config.json \
		${CONFIG_DIR}/broadtwitter_config.json \
		${CONFIG_DIR}/casie_config.json \
		${CONFIG_DIR}/conll03_config.json \
		${CONFIG_DIR}/crossner_ai_config.json \
		${CONFIG_DIR}/crossner_literature_config.json \
		${CONFIG_DIR}/crossner_music_config.json \
		${CONFIG_DIR}/crossner_politics_config.json \
		${CONFIG_DIR}/crossner_science_config.json \
		${CONFIG_DIR}/diann_config.json \
		${CONFIG_DIR}/fabner_config.json \
		${CONFIG_DIR}/fewrel_config.json \
		${CONFIG_DIR}/harveyner_config.json \
		${CONFIG_DIR}/mitmovie_config.json \
		${CONFIG_DIR}/mitrestaurant_config.json \
		${CONFIG_DIR}/ncbidisease_config.json \
		${CONFIG_DIR}/ontonotes_config.json \
		${CONFIG_DIR}/rams_config.json \
		${CONFIG_DIR}/wnut17_config.json \
	--output "data/processed_w_BES_RES" \
	--overwrite_output_dir \
	--include_examples \
	--top_k ${TOP_K} \
	--device "cuda:0"
```

## Training
```
bash scripts/train.sh
```

Content of the bash file
```bash
#!/bin/bash

source ~/.anaconda3/bin/activate
conda activate GoLLIE

if [ -f .env ]; then
export $(cat .env | grep -v '^#' | xargs)
fi

export CUDA_VISIBLE_DEVICES=0


torchrun --nproc_per_node 1 --master_port 0 src/run.py ./configs/train_random_RES.json
```

`torchrun`: Used to control running code with DDP (Distributed Data Parallel)

`nproc_per_node`: Used to control the number of GPUs participating in parallel processing

`src/run.py`: Main file

`./configs/train_SLR.json`: Specific experiment configuration; different experiments correspond to different JSON files

## Inference
```
bash scripts/inference.sh
```

Content of the bash file
```bash
#!/bin/bash

source ~/.anaconda3/bin/activate
conda activate GoLLIE

if [ -f .env ]; then
export $(cat .env | grep -v '^#' | xargs)
fi

CUDA_VISIBLE_DEVICES="0" python src/run.py ./configs/inference_ReNIE_random_RES.json
```
Inference runs on a single GPU.

Use configuration files starting with `inference`.

## Evaluation (Calculate F1 Score, including precision and recall)
```
bash scripts/evaluate.sh
```

Content of the bash file
```bash
#!/bin/bash

source ~/.anaconda3/bin/activate
conda activate GoLLIE

if [ -f .env ]; then
export $(cat .env | grep -v '^#' | xargs)
fi

CUDA_VISIBLE_DEVICES="0" python './src/evaluate.py' './configs/evaluate.json'
```

`CUDA_VISIBLE_DEVICES`: Controls which GPU to use

`./src/evaluate.py`: Main function entry point

`./configs/evaluate.json`: Configuration file, used to instantiate HfArgumentParser

# Configuration Files
All configuration files are located in `./configs`. For detailed explanations of each field in the configuration files, please refer to `./src/arguments.py`.
