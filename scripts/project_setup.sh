#!/bin/bash

# Script Name: env_setup.sh
# Description: Setup Python environment and dependencies.
#   !Important Note! This script is mean to be used with conda as venv manager.
#
# Usage: ./env_setup.sh

conda create --name recurrent_homer python=3.11
conda activate recurrent_homer

pip install poetry
poetry install