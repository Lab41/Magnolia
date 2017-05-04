#!/bin/bash

# Set up dependencies and install one version of the magnolia environment

set -e

if [[ $OSTYPE == darwin* ]]; then
    TF_URL_GPU="https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-1.1.0-py3-none-any.whl"
    TF_URL_CPU="https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0-py3-none-any.whl"
else
    TF_URL_GPU="https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl"
    TF_URL_CPU="https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp35-cp35m-linux_x86_64.whl"
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

while [[ $# -ge 1 ]]
do
key="$1"

case $key in
    --gpu|-g)
    GPU="true"
    if [[ "$TF_URL" == "" ]]; then
        TF_URL="$TF_URL_GPU"
    fi
    ;;
    --tf-url|-t)
    TF_URL="$2"
    shift
    ;;
    *)
    echo "install.sh -- Install dependencies, conda environment and package for "
    echo " Magnolia project."
    echo
    echo "Options:"
    printf " --gpu|-g\t\tInstall GPU version of tensorflow\n"
    printf " --tf-url|-t URL\t\tInstall from this Tensorflow URL (1.1 by default)\n\n"
    exit
            # unknown option
    ;;
esac
shift # past argument or value
done

# Fall back to CPU tensorflow
if [[ "$TF_URL" == "" ]]; then
    TF_URL="$TF_URL_CPU"
fi

if [[ "$GPU" == "true" ]]; then
    conda create -f "$DIR"/environment-gpu.yml
    # ungodly hack here (get path prefix for new environment)
    ENV_PREFIX=$(conda info --envs --json | python -c 'import json; import sys; print("\n".join([env for env in json.loads(sys.stdin.read())["envs"] if "magnolia3-gpu" in env]))')
    echo "New environment: $ENV_PREFIX"
else
    conda env create --force -f "$DIR"/environment-cpu.yml

    # ungodly hack here (get path prefix for new environment)
    ENV_PREFIX=$(conda info --envs --json | python -c 'import json; import sys; print("\n".join([env for env in json.loads(sys.stdin.read())["envs"] if "magnolia3-cpu" in env]))')
    echo "New environment: $ENV_PREFIX"
fi

# Install tensorflow
"$ENV_PREFIX"/bin/pip install --upgrade "$TF_URL"

# Install Magnolia
"$ENV_PREFIX"/bin/pip install --upgrade --no-deps "$DIR"
