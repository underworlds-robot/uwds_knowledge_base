#!/bin/bash
#
echo "Start downloding the Glove dataset... this may take a while."
cd ./res
mkdir data
cd data
mkdir glove && cd glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip
cd ..
echo "Start downloding the Geo880 dataset... this may take a while."
git clone https://github.com/mllovers/geo880-sparql.git
cd ../..
echo "Bye bye !"
