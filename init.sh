#!/bin/sh

# Official Dataset

echo "* Download the kaggle competitions dataset: kaggle-llm-science-exam"
kaggle competitions download -c kaggle-llm-science-exam && \
unzip -o kaggle-llm-science-exam.zip -d data/kaggle-llm-science-exam
rm -f kaggle-llm-science-exam.zip


echo "* Download the Wiki 270k STEM dataset: all-paraphs-parsed-expanded"
kaggle datasets download -d mbanaei/all-paraphs-parsed-expanded && \
unzip -o all-paraphs-parsed-expanded.zip -d data/all-paraphs-parsed-expanded
rm -f all-paraphs-parsed-expanded.zip

echo "* Download the Old Wiki 270k STEM dataset: stem-wiki-cohere-no-emb"
kaggle datasets download -d mbanaei/stem-wiki-cohere-no-emb && \
unzip -o stem-wiki-cohere-no-emb.zip -d data/stem-wiki-cohere-no-emb
rm -f stem-wiki-cohere-no-emb.zip

echo "* Download additional dataset: ranchantan/llm-se-datasets-with-context-v2"
kaggle datasets download -d ranchantan/llm-se-datasets-with-context-v2 && \
unzip -o llm-se-datasets-with-context-v2.zip -d data/llm-se-datasets-with-context-v2
rm -f llm-se-datasets-with-context-v2.zip

echo "* Download additional dataset: ranchantan/llm-se-extra-train-datasets"
kaggle datasets download -d ranchantan/llm-se-extra-train-datasets && \
unzip -o llm-se-extra-train-datasets.zip -d data/llm-se-extra-train-datasets
rm -f llm-se-extra-train-datasets.zip

echo "* Download additional dataset: ranchantan/llm-se-datasets-with-context-v3"
kaggle datasets download -d ranchantan/llm-se-datasets-with-context-v3 && \
unzip -o llm-se-datasets-with-context-v3.zip -d data/llm-se-datasets-with-context-v3
rm -f llm-se-datasets-with-context-v3.zip

echo "* Download additional dataset: ranchantan/llm-se-datasets-with-context-v4"
kaggle datasets download -d ranchantan/llm-se-datasets-with-context-v4 && \
unzip -o llm-se-datasets-with-context-v4.zip -d data/llm-se-datasets-with-context-v4
rm -f llm-se-datasets-with-context-v4.zip






echo "* Download the faiss-index of the Wiki 270k STEM dataset: bge-small-faiss"
kaggle datasets download -d simjeg/bge-small-faiss && \
unzip -o bge-small-faiss.zip -d data/bge-small-faiss
rm -f bge-small-faiss


# Llama2 7B

echo "* Download llama2-pretrained dataset: lizhecheng/llama2-7b-hf"
kaggle datasets download -d lizhecheng/llama2-7b-hf && \
unzip -o llama2-7b-hf.zip -d data/
rm -f llama2-7b-hf.zip

# Llama2 13B

mkdir -p data/weyaxi/

echo "* Download llama2-pretrained dataset: weyaxi/nousresearch"
kaggle datasets download -d weyaxi/nousresearch && \
unzip -o nousresearch.zip -d data/weyaxi/llama2-13b
rm -f nousresearch.zip

# for OpenBook

echo "* Download sentence-transformer-pretrained dataset for OpenBook: yonaschanie/sentencetransformers-allminilml6v2"
kaggle datasets download -d yonaschanie/sentencetransformers-allminilml6v2 && \
unzip -o sentencetransformers-allminilml6v2.zip -d data/sentencetransformers-allminilml6v2
rm -f sentencetransformers-allminilml6v2.zip

echo "* Download wikipedia-faiss-index dataset for OpenBook: jjinho/wikipedia-2023-07-faiss-index"
kaggle datasets download -d jjinho/wikipedia-2023-07-faiss-index && \
unzip -o wikipedia-2023-07-faiss-index.zip -d data/wikipedia-2023-07-faiss-index
rm -f wikipedia-2023-07-faiss-index.zip

echo "* Download wikipedia-plaintext dataset for OpenBook: jjinho/wikipedia-20230701"
kaggle datasets download -d jjinho/wikipedia-20230701 && \
unzip -o wikipedia-20230701.zip -d data/wikipedia-20230701
rm -f wikipedia-20230701.zip

# Additional Dataset

# Additional cdeotte Dataset

mkdir -p data/llm-se-extra-train-datasets/cdeotte

echo "* Download additional dataset: cdeotte/60k-data-with-context-v2"
kaggle datasets download -d cdeotte/60k-data-with-context-v2 && \
unzip -o 60k-data-with-context-v2.zip -d data/llm-se-extra-train-datasets/cdeotte/60k-data-with-context-v2
rm -f 60k-data-with-context-v2.zip

echo "* Download additional dataset: cdeotte/40k-data-with-context-v2"
kaggle datasets download -d cdeotte/40k-data-with-context-v2 && \
unzip -o 40k-data-with-context-v2.zip -d data/llm-se-extra-train-datasets/cdeotte/40k-data-with-context-v2
rm -f 40k-data-with-context-v2.zip


# Additional radek1 Datasets

mkdir -p data/llm-se-extra-train-datasets/radek1

echo "* Download additional dataset: radek1/additional-train-data-for-llm-science-exam"
kaggle datasets download -d radek1/additional-train-data-for-llm-science-exam && \
unzip -o additional-train-data-for-llm-science-exam.zip -d data/llm-se-extra-train-datasets/radek1/additional-train-data-for-llm-science-exam
rm -f additional-train-data-for-llm-science-exam.zip


echo "* Download additional dataset: radek1/15k-high-quality-examples"
kaggle datasets download -d radek1/15k-high-quality-examples && \
unzip -o 15k-high-quality-examples.zip -d data/llm-se-extra-train-datasets/radek1/15k-high-quality-examples
rm -f 15k-high-quality-examples.zip


echo "* Download additional dataset: radek1/sci-or-not-sci-hypthesis-testing-pack"
kaggle datasets download -d radek1/sci-or-not-sci-hypthesis-testing-pack && \
unzip -o sci-or-not-sci-hypthesis-testing-pack.zip -d data/llm-se-extra-train-datasets/radek1/sci-or-not-sci-hypthesis-testing-pack
rm -f sci-or-not-sci-hypthesis-testing-pack.zip

# Additional leonidkulyk Datasets

mkdir -p data/llm-se-extra-train-datasets/leonidkulyk

echo "* Download additional dataset: leonidkulyk/wikipedia-stem-1k"
kaggle datasets download -d leonidkulyk/wikipedia-stem-1k && \
unzip -o wikipedia-stem-1k.zip -d data/llm-se-extra-train-datasets/leonidkulyk/wikipedia-stem-1k
rm -f wikipedia-stem-1k.zip


echo "* Download additional wiki-stem-articles dataset: ranchantan/llm-se-additional-wiki-stem-articles"
kaggle datasets download -d ranchantan/llm-se-additional-wiki-stem-articles && \
unzip -o llm-se-additional-wiki-stem-articles.zip -d data/llm-se-additional-wiki-stem-articles
rm -f llm-se-additional-wiki-stem-articles.zip

# Additional yalickj Datasets

mkdir -p data/llm-se-extra-train-datasets/yalickj

kaggle datasets download -d yalickj/dataset-wiki-new-1 && \
unzip -o dataset-wiki-new-1.zip -d data/llm-se-extra-train-datasets/yalickj/dataset-wiki-new-1
rm -f dataset-wiki-new-1.zip


mkdir -p data/llm-se-extra-train-datasets/wuwenmin

kaggle datasets download -d wuwenmin/llm-sci-eval300-gpt4-corrected && \
unzip -o llm-sci-eval300-gpt4-corrected.zip -d data/llm-se-extra-train-datasets/wuwenmin/llm-sci-eval300-gpt4-corrected
rm -f llm-sci-eval300-gpt4-corrected.zip

mkdir -p data/llm-se-extra-train-datasets/takeshisuzuki

kaggle datasets download -d takeshisuzuki/additional-dataset-800articles-4000rows && \
unzip -o additional-dataset-800articles-4000rows.zip -d data/llm-se-extra-train-datasets/takeshisuzuki/additional-dataset-800articles-4000rows
rm -f additional-dataset-800articles-4000rows.zip


kaggle datasets download -d ranchantan/kaggle-llm-science-exam-with-context && \
unzip -o kaggle-llm-science-exam-with-context.zip -d data/kaggle-llm-science-exam-with-context/
rm -f kaggle-llm-science-exam-with-context.zip
