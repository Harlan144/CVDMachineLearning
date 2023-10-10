#! /bin/bash

set -o errexit

docker build -t harlan144/cvd_test .

#Run commands interactively
#dockerCommand="docker run -i -t --rm --user $(id -u):$(id -g) -v $(pwd):/sandbox -v /tmp:/tmp --workdir=/sandbox harlan144/cvd_test"

#Uncomment this line to run commands in detached mode
dockerCommand="docker run -d -t --rm --user $(id -u):$(id -g) -v $(pwd):/sandbox -v /tmp:/tmp --workdir=/sandbox harlan144/cvd_test"

#To run any python files, run ./main.sh {name_of_file}. For example, "./main.sh saveModel_no_transfer.py"

$dockerCommand python3 $1
