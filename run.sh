#!/bin/bash
# unitrace \
#   --chrome-kernel-logging \
#   --start-paused \
#   --output-dir-path artifacts/molmoact2 \
#   python collect_model.py

python -m perf_analysis \
  artifacts/molmoact2/collection.json \
  --output-dir artifacts/molmoact2 --allow-profiler-fallback \
  --unitrace artifacts/molmoact2/python.1119469.json