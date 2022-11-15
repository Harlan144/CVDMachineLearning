#! /bin/bash

set -o errexit

docker build -t harlan144/cvd_test .

dockerCommand="docker run -i -t --rm --user $(id -u):$(id -g) -v $(pwd):/sandbox -v /tmp:/tmp --workdir=/sandbox harlan144/cvd_test"

$dockerCommand python3 BioImageAnalysis/machineLearning.py


