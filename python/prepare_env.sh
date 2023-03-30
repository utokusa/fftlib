#!/bin/bash

# Usage: Run "./activate.sh" then "source env/bin/activate"
# Stop using python: deactivate

set -e

ENV_NAME=env
SCRIPT_DIR=$(cd $(dirname $0); pwd)
PYTHON_DIR=$SCRIPT_DIR/$ENV_NAME
if [ ! -f "env/bin/activate" ]; then
    python3 -m venv $PYTHON_DIR
fi

source $PYTHON_DIR/bin/activate

pip install -U  pip
pip install -U -r $SCRIPT_DIR/requirements.txt

