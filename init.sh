#!/bin/sh

echo "* Download the kaggle competitions dataset: kaggle-llm-science-exam"
kaggle competitions download -c kaggle-llm-science-exam && \
unzip -o kaggle-llm-science-exam.zip -d data/kaggle-llm-science-exam
rm -f kaggle-llm-science-exam.zip

# Llama2 7B

echo "* Download llama2-pretrained dataset: lizhecheng/llama2-7b-hf"
kaggle datasets download -d lizhecheng/llama2-7b-hf && \
unzip -o llama2-7b-hf.zip -d data/
rm -f llama2-7b-hf.zip

# Llama2 7B

echo "* Download llama2-pretrained dataset: weyaxi/nousresearch"
kaggle datasets download -d weyaxi/nousresearch && \
unzip -o nousresearch.zip -d data/weyaxi/llama2-13b
rm -f nousresearch.zip

# additional radek1 datasets

mkdir -p data/llm-se-extra-train-datasets/radek1

echo "* Download additional dataset: radek1/additional-train-data-for-llm-science-exam"
kaggle datasets download -d radek1/additional-train-data-for-llm-science-exam && \
unzip -o additional-train-data-for-llm-science-exam.zip -d data/llm-se-extra-train-datasets/radek1/additional-train-data-for-llm-science-exam
rm -f additional-train-data-for-llm-science-exam.zip


echo "* Download additional dataset: radek1/15k-high-quality-examples"
kaggle datasets download -d radek1/15k-high-quality-examples && \
unzip -o 15k-high-quality-examples.zip -d data/llm-se-extra-train-datasets/radek1/15k-high-quality-examples
rm -f 15k-high-quality-examples.zip

# additional leonidkulyk datasets

mkdir -p data/llm-se-extra-train-datasets/leonidkulyk

echo "* Download additional dataset: leonidkulyk/wikipedia-stem-1k"
kaggle datasets download -d leonidkulyk/wikipedia-stem-1k && \
unzip -o wikipedia-stem-1k.zip -d data/llm-se-extra-train-datasets/leonidkulyk/wikipedia-stem-1k
rm -f wikipedia-stem-1k.zip
