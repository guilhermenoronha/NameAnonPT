# Name Entity Recognition for Brazilian Portuguese.

The present repository is a study of Name Entity Recognition application to detect proper nouns in Portuguese.

The study was carried on [CE-DOHS corpus](http://www.tycho.iel.unicamp.br/cedohs/corpora.html) (Corpus Eletrônico de Documentos Históricos do Sertão). 

CE-DOHS was preprocessed and, later, annotated using [label-studio](https://labelstud.io/). This study aimed to calculate the NER F-Score using [BI-LSTM-CRF](https://pypi.org/project/bi-lstm-crf/) deep learning algorithm.

All the process can be found in Full Pipeline notebook.

The F-Score obtained was *0.97*

## How to use: Notebook version

1. conda create -n NameAnonPT
2. conda activate NameAnonPT
3. pip install -r requirements.txt
4. jupyter lab

## How to use: Airflow Version

OBS: this version requires Docker and Docker-Compose.

1. Go to Airflow folder
2. docker-compose up
3. Access localhost:/8080
4. run cartasAnonPT DAG

Below some screenshots:

![DAG](https://user-images.githubusercontent.com/2208226/161294508-98ceefa8-a025-4590-a7c4-7deca5dde1cb.png)
![tree](https://user-images.githubusercontent.com/2208226/161294515-af9e6101-30ce-434b-9deb-2dfac4790ee1.png)
