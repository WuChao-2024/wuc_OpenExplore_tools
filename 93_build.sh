#!/bin/bash

set -e -v
cd $(dirname $0)
config_file="config.yaml"
model_type="onnx"
# build model
hb_mapper makertbin --config ${config_file}  \
                    --model-type  ${model_type}
chmod 777 ./*
chmod 777 ./bin_dir/*
