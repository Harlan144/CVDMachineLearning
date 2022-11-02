#! /bin/bash

set -o errexit

docker build -t harlan144/cvd_test .

mkdir -p data

dockerCommand="docker run -i -t --rm --user $(id -u):$(id -g) -v $(pwd):/sandbox -v $(pwd)/data:/data -v /tmp:/tmp --workdir=/sandbox harlan144/cvd_test"

$dockerCommand python3 classifycvd0.py
