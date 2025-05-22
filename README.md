# 环境安装
```bash
conda create -n GoLLIE python==3.9.*
bash requirements.sh
conda activate GoLLIE
```

# 运行代码
## 生成数据
```
bash scripts/generate_data.sh
```

bash文件内容
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


## 训练部分
```
bash scripts/train.sh
```

bash文件内容
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

`torchrun`: 用来控制使用DDP运行代码

`nproc_per_node`: 用来控制参与并行的GPU数量

`src/run.py`: 主文件

`./configs/train_SLR.json`: 具体的实验配置, 不同的实验对应不同的json文件

## 推理部分
```
bash scripts/inference.sh
```

bash文件内容
```bash
#!/bin/bash

source ~/.anaconda3/bin/activate
conda activate GoLLIE

if [ -f .env ]; then
export $(cat .env | grep -v '^#' | xargs)
fi

CUDA_VISIBLE_DEVICES="0" python src/run.py ./configs/inference_ReNIE_random_RES.json
```
推理部分使用单卡跑代码

配置文件使用 `inference` 开头的配置文件

## 评估 (计算F1 Score, 包括precision和recall)
```
bash scripts/evaluate.sh
```

bash文件的内容
```bash
#!/bin/bash

source ~/.anaconda3/bin/activate
conda activate GoLLIE

if [ -f .env ]; then
export $(cat .env | grep -v '^#' | xargs)
fi

CUDA_VISIBLE_DEVICES="0" python './src/evaluate.py' './configs/evaluate.json'
```

`CUDA_VISIBLE_DEVICES`: 控制使用第几块GPU

`./src/evaluate.py`: 主函数入口

`./configs/evaluate.json`: 配置文件, 用来实例化HfArgparser


# 配置文件
所有配置文件都存放于 `./configs` 中, 配置文件每个字段的含义详见 `./src/arguments.py`