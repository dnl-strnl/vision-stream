#!/bin/bash

TASK=detect
VERSION=yolov8s
VARIANT=worldv2
MODEL=${VERSION}-${VARIANT}

poetry run torch-model-archiver \
--model-name=$MODEL \
--handler=src/vision_stream/models/yolo/$TASK.py \
--serialized-file=models/$MODEL.pt \
--runtime=python3 --version 1 --force

TIMESTAMP=$(date +"%m%d%Y")
MODEL_FILE=${MODEL}-${TIMESTAMP}.mar

mv $MODEL.mar models/$MODEL_FILE
MODEL_ARG=models/$MODEL_FILE

md5sum=$(md5 -q ${MODEL_ARG})

echo "md5sum: ${md5sum}"
echo "poetry run torchserve --start --ncs --model-store=models/ --models $MODEL_FILE"
