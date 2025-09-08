#!/bin/bash
set -e

## Running Research Question 1 Experiments:
bash run_opt.sh

## Running Research Question 3 Experiments:
bash run_fs.sh

## Analyzing RQ1 and RQ3 Results:
python3.12 ../src/post_exp.py

## Experiments for Research Question 2:
python3.12 ../src/explain.py ../datasets/moot/optimize/process/coc1000.csv