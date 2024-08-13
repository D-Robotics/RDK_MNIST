#!/usr/bin/env sh

mkdir -p mnist_raw
cd mnist_raw
wget https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

echo "MNIST data downloaded and unzipped."

