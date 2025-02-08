#!/usr/bin/env sh

set -e -v
cd $(dirname $0) || exit

model_type="onnx"
onnx_model="pt/yolo11n.onnx"
march="bayes-e"

hb_mapper checker --model-type ${model_type} \
                  --model ${onnx_model} \
                  --march ${march}

chmod 777 ./*
