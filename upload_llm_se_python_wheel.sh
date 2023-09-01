#!/bin/sh

python3 -m build
kaggle datasets version -p dist/ -m "" -r skip

#python3 -m pip download --only-binary :all: -r requirements.txt -d dist/wheels
#kaggle datasets version -p dist/wheels -m "" -r skip
