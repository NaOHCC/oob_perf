#!/bin/bash

DEVICE_TYPE=$1
DEVICE_NAME=$2

python collect_model.py \
  --device "$DEVICE_TYPE" \
  --output-dir "artifacts/molmoact2/$DEVICE_NAME" \
  --hardware-label "A100 BF16" \
  --peak-tflops 260 \
  --memory-bandwidth-gbs 1555

python -m perf_analysis \
  artifacts/molmoact2/"$DEVICE_NAME"/collection.json \
  --output-dir artifacts/molmoact2/"$DEVICE_NAME"