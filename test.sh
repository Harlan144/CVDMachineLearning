#! /bin/bash

set -o errexit

docker build -t harlan144/cvd_test .

#Run docker in interactive mode
dockerCommand="docker run -i -t --rm --user $(id -u):$(id -g) -v $(pwd):/sandbox -v /tmp:/tmp --workdir=/sandbox harlan144/cvd_test"

$dockerCommand python3 BioImageAnalysis/barGraph.py


