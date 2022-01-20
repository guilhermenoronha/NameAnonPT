from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from bi_lstm_crf.app import WordsTagger
import glob
import os
import re

def _load_txt():
    file_path = glob.glob('/opt/airflow/dags/corpus/*.txt')
    files = []
    for file in file_path:
        with open(file, 'r', encoding='UTF-8') as f:
            files.append(f.read())
    return files

def _file_to_dict(ti):
    dicts = []
    files = ti.xcom_pull(key='return_value', task_ids=['Load_Files'])
    for file in files[0]:
        lines  = file.splitlines()
        dicts.append({
            'corpus': lines[1].replace('|:|', '').strip(),
            'author': lines[2].replace('|:|', '').strip(),
            'source': lines[4].replace('|:|', '').strip(),
            'sender': lines[5].replace('|:| Autor: ', '').strip(),
            'recipient': lines[6].replace('|:| Destinatário: ', '').strip(),
            'datetime': lines[7].replace('|:| Data: ', '').strip(),
            'version': lines[8].replace('|:|', '').strip(),
            'encoding': lines[9].replace('|:| Encoding: ', '').strip(),
            'letter': lines[11].replace('Carta ', '').strip(),
            'letter condition': lines[13].strip(),
            'text': ''.join(lines[15:]),
            'preprocessed text': re.sub('\[.+?\]', '', ''.join(lines[15:]))
        })
    return dicts

def _write_label_studio(ti):
    dicts = ti.xcom_pull(key='return_value', task_ids= ['File_To_Dict'])
    for dict_ in dicts[0]:
        print('dict: ',dict_.get('letter'))
        with open (f'/opt/airflow/dags/label-studio/{dict_.get("letter")}.txt', 'w', encoding='UTF-8') as f:
            f.write(dict_.get('preprocessed text'))


def _append_conll_to_dicts(ti):
    dicts = ti.xcom_pull(key='return_value', task_ids= ['File_To_Dict'])[0][:]
    text = []
    text_list = []
    tags = []
    tag_list = []

    with open('/opt/airflow/dags/annotated-corpus/annotated-corpus.conll', 'r', encoding='UTF-8') as f:
        full_text = f.readlines()

    for line in full_text[1:]:
        if line != '\n':
            words = line.split()
            tags.append(f'{words[3]}')
            if words[0] == '"':
                text.append(f'\\{words[0]}')
            elif words[0] == '\\':
                text.append(f'\\{words[0]}')
            else:
                text.append(f'{words[0]}')
        else:
                text_list.append(text)
                tag_list.append(tags)
                text = []
                tags = []

    for i, dict_ in enumerate(dicts):
        preprop_array = dict_.get('preprocessed text').split()
        for j, text in enumerate(text_list):
            if preprop_array[:5] == text[:5]:
                dicts[i]['tokens'] = text_list[j]
                dicts[i]['tags'] = tag_list[j]
    return dicts

def _ingest_data(ti):
    dicts = ti.xcom_pull(key='return_value', task_ids =['Append_CONLL_to_dicts'])[0][:]
    client = MongoClient('mongo', 27017, username='root', password='example')
    db = client['Letters']
    collection = db['cartasdosertao']
    if (collection.count() == 0):
        collection.insert_many(dicts)

def _create_test_train_data(ti):
    dicts = ti.xcom_pull(key='return_value', task_ids =['Append_CONLL_to_dicts'])[0][:]
    X_train, X_test = train_test_split(dicts, test_size=0.20, random_state=42)

    # bilstm-crf library requires the train dataset to be named as dataset.txt
    with open ('/opt/airflow/dags/test_train_data/dataset.txt', 'w', encoding='UTF-8') as f:
            for document in X_train:
                f.write('[%s]\t[%s]\n' % (', '.join(f'"{token}"' for token in document.get('tokens')),
                                        ', '.join(f'"{tag}"' for tag in document.get('tags'))))

    with open ('/opt/airflow/dags/test_train_data/test.txt', 'w', encoding='UTF-8') as f:
            for document in X_test:
                f.write('[%s]\t[%s]\n' % (', '.join(f'"{token}"' for token in document.get('tokens')),
                                        ', '.join(f'"{tag}"' for tag in document.get('tags'))))
    ti.xcom_push(key='X_train', value=X_train)
    ti.xcom_push(key='X_test', value=X_test)

def _create_vocab_and_tags_files(ti):
    tags = ('B-PER', 'I-PER', 'O')
    with open('/opt/airflow/dags/test_train_data/tags.json', 'w', encoding='UTF-8') as f:
        f.write('[%s]' % (', '.join(f'"{tag}"' for tag in tags)) )
        
    vocab = []
    X_train = ti.xcom_pull(key='X_train', task_ids=['Create_Test_Train_Data'])[0][:]
    for doc in X_train:
        vocab.extend(doc.get('tokens'))
    vocab = list(dict.fromkeys(vocab))
    with open('/opt/airflow/dags/test_train_data/vocab.json', 'w', encoding='UTF-8') as f:
        f.write('[%s]' % (', '.join(f'"{token}"' for token in vocab)) )
    return tags
    
def _test_model(ti):
    def tags_to_numbers(taglist, tags):
        res = []
        for tag in tags:
            for i, t in enumerate(taglist):
                if t == tag:
                    res.append(i)
        return res
    
    test_tokens = []
    test_tags = []
    X_test = ti.xcom_pull(key='X_test', task_ids=['Create_Test_Train_Data'])[0][:]

    for doc in X_test:
        test_tokens.append(doc.get('tokens'))
        test_tags.append(doc.get('tags'))

    y_true = []
    y_pred = []
    tags = ti.xcom_pull(key='return_value', task_ids=['Create_Vocab__And_Tags_Files'])[0][:]
    for i, token in enumerate(test_tokens):
        true_tags = tags_to_numbers(tags, test_tags[i])
        y_true += true_tags
        model = WordsTagger(model_dir='/opt/airflow/dags/cartas_anon_model')
        pred_tags, _ = model([token])
        pred_tokens = tags_to_numbers(tags, pred_tags[0])
        y_pred += pred_tokens

    f1 = f1_score(y_true, y_pred, average='micro')
    matrix = confusion_matrix(y_true, y_pred)
    with open('/opt/airflow/dags/results.txt', 'w', encoding='UTF-8') as f:
        f.write(f'Total tokens: {len(y_true)}')
        f.write(f'Total B-NOME: {y_true.count(0)}\n')
        f.write(f'Total I-NOME: {y_true.count(1)}\n')
        f.write(f'Total O: {y_true.count(2)}\n')
        f.write(f'F-Score = {f1}\n')
        f.write(f'Confusion Matrix: {matrix}')

def _is_conll_created():
    if not os.path.isfile('/opt/airflow/dags/annotated-corpus/annotated-corpus.conll'):
        raise ValueError('File doesn\'t exist.')


default_args = {
    'owner': 'Guilherme Noronha',
    'start_date': days_ago(1),
    'retries': 5,
    'retry_delay': timedelta(minutes=1)
}

with DAG(dag_id = 'cartasAnonPT', 
         schedule_interval='@once', 
         default_args=default_args,
         catchup=False) as dag:
    
    t1 = PythonOperator(
        task_id = 'Load_Files',
        python_callable = _load_txt
    )

    t2 = PythonOperator(
        task_id = 'File_To_Dict',
        python_callable = _file_to_dict
    )

    t3 = BashOperator(
        task_id = 'Create_LabelStudio_Folder',
        bash_command = 'mkdir -p -m 777 /opt/airflow/dags/label-studio'
    )

    t4 = PythonOperator(
        task_id = 'Write_Label-Studio_Files',
        python_callable = _write_label_studio
    )

    t5 = PythonOperator(
        task_id = 'Append_CONLL_to_dicts',
        python_callable = _append_conll_to_dicts
    )

    t6 = PythonOperator(
        task_id = 'Ingest_Data',
        python_callable= _ingest_data
    )

    t7 = BashOperator(
        task_id = 'Create_Test_Train_Folder',
        bash_command = 'mkdir -p -m 777 /opt/airflow/dags/test_train_data'
    )

    t8 = PythonOperator(
        task_id = 'Create_Test_Train_Data',
        python_callable = _create_test_train_data
    )

    t9 = PythonOperator(
        task_id = 'Create_Vocab__And_Tags_Files',
        python_callable= _create_vocab_and_tags_files
    )

    t10 = BashOperator(
        task_id = 'Run_BILSTM_Learner',
        bash_command = 'python -m bi_lstm_crf "/opt/airflow/dags/test_train_data" --model_dir "/opt/airflow/dags/cartas_anon_model" --num_epoch 500 --save_best_val_model'
    )

    t11 = PythonOperator(
        task_id = 'Test_Model',
        python_callable = _test_model
    )

    t12 = PythonOperator(
        task_id = 'Is_CONLL_Created',
        python_callable = _is_conll_created
    )

    t1 >> t2
    [t2, t12] >> t5 >> t6
    [t2, t3] >> t4 >> t12
    [t7, t5] >> t8 >> t9 >> t10 >> t11    