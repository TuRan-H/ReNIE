#!/bin/bash

ACE_DATA_PATH="./download/ace2005/ace_2005_td_v7/data"
ACE_DATA_SPLITS="./download/ace2005/splits/english"
LANGUAGE="english" # "chinese"
OUTPUT_PATH="./download/ace2005/ace05"

mkdir ${OUTPUT_PATH}

python src/tasks/ace/preprocess_ace.py \
    -i ${ACE_DATA_PATH} \
    -o ${OUTPUT_PATH} \
    -s ${ACE_DATA_SPLITS} \
    -l ${LANGUAGE} \
    --time_and_val
