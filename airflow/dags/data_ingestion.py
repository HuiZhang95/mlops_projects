from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

from evidently.report import Report
from evidently.metric_preset import ClassificationPreset

from datetime import datetime
import requests
import os
import json
import pandas as pd
import numpy as np
import sys
# from logger import logger
import logging
import glob
import shutil
import joblib
import subprocess

# Adjust sys.path to include the 'project' directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_dir)

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.insert(0, code_dir)

print("how to print?")

print(f"Current working directory: {os.getcwd()}")

# from src.config.config import Config
# from src.config.check_structure import check_existing_file, check_existing_folder

itemNoAll = 100

input_filepath = "./data/new_data/old/"#Config.NEW_DATA_BACKUP_DIR
output_filepath = "./data/new_data/"#Config.NEW_DATA_DIR

old_filepath = "./data/raw/"
processed_folder = "./data/processed/"
base_joblib_filename = 'model_best_rf'
os.makedirs(input_filepath, exist_ok=True)
os.makedirs(output_filepath, exist_ok=True)

logging.info(f"Current working directory: {os.getcwd()}")
print(f"Current working directory: {os.getcwd()}")

def generate_new_data():

    os.makedirs(output_filepath, exist_ok=True)

    print(f"Current working directory: {os.getcwd()}")

    # Prompt the user for input file paths
    # input_filepath= click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    input_filepath_users = os.path.join(input_filepath, "usagers-2023.csv")
    input_filepath_caract = os.path.join(input_filepath, "caract-2023.csv")
    input_filepath_places = os.path.join(input_filepath, "lieux-2023.csv")
    input_filepath_veh = os.path.join(input_filepath, "vehicules-2023.csv")

    #--Importing dataset
    df_users = pd.read_csv(input_filepath_users, sep=";")
    df_caract = pd.read_csv(input_filepath_caract, sep=";", header=0, low_memory=False)
    df_places = pd.read_csv(input_filepath_places, sep = ";", encoding='utf-8')
    df_veh = pd.read_csv(input_filepath_veh, sep=";")

    # print(df_users.head(), len(df_users), df_caract.head(), len(df_caract), df_places.head(), len(df_places), df_veh.head(), len(df_veh))
    # randomly select 100 items
    allChoices = df_users['Num_Acc'].unique()

    selected_items = np.random.choice(allChoices, size=itemNoAll, replace=False)
    # print(selected_items)

    df_users_1 = df_users[df_users['Num_Acc'].isin(selected_items)]
    df_caract_1 = df_caract[df_caract['Num_Acc'].isin(selected_items)]
    df_places_1 = df_places[df_places['Num_Acc'].isin(selected_items)]
    df_veh_1 = df_veh[df_veh['Num_Acc'].isin(selected_items)]

    for file, filename in zip([df_users_1, df_caract_1, df_places_1, df_veh_1], ['usagers', 'caracteristiques', 'lieux', 'vehicules']):
        current_date = datetime.now().strftime('%Y%m%d')
        output_file = os.path.join(output_filepath, f'{filename}-{current_date}.csv')
        file.to_csv(output_file, index=False)

def check_new_csv(folder_path):
    csv_files = glob.glob(f"{folder_path}/*.csv")
    if csv_files:
        return csv_files
    else:
        return False

def remove_new_csv(csv_files):
    # Ensure destination folder exists
    os.makedirs(output_filepath, exist_ok=True)
    for file in csv_files:
        shutil.move(file, output_filepath)
    
def merge_old_new():
    input_filepath_users = os.path.join(old_filepath, "usagers-2021.csv")
    input_filepath_caract = os.path.join(old_filepath, "caracteristiques-2021.csv")
    input_filepath_places = os.path.join(old_filepath, "lieux-2021.csv")
    input_filepath_veh = os.path.join(old_filepath, "vehicules-2021.csv")

    #--Importing dataset
    df_users = pd.read_csv(input_filepath_users, sep=";", on_bad_lines='skip')
    df_caract = pd.read_csv(input_filepath_caract, sep=";", header=0, low_memory=False, on_bad_lines='skip')
    df_places = pd.read_csv(input_filepath_places, sep = ";", encoding='utf-8', on_bad_lines='skip')
    df_veh = pd.read_csv(input_filepath_veh, sep=";", on_bad_lines='skip')

    csv_files = check_new_csv(output_filepath)

    if csv_files:
        for csv_file in csv_files:
            tmp_file = csv_file
            if csv_file[0:3] == 'usa':
                tmp = pd.read_csv(tmp_file, sep=";", on_bad_lines='skip')
                df_users = pd.concat([df_users, tmp], ignore_index=True)
            elif csv_file[0:3] == 'car':
                tmp = pd.read_csv(tmp_file, sep=";", header=0, low_memory=False, on_bad_lines='skip')
                df_caract = pd.concat([df_caract, tmp], ignore_index=True)
            elif csv_file[0:3] == 'lie':
                tmp = pd.read_csv(tmp_file, sep=";", encoding='utf-8', on_bad_lines='skip')
                df_places = pd.concat([df_places, tmp], ignore_index=True)
            elif csv_file[0:3] == 'veh':
                tmp = pd.read_csv(tmp_file, sep=";", on_bad_lines='skip')
                df_veh = pd.concat([df_veh, tmp], ignore_index=True)
        
        for file, filename in zip([df_users, df_caract, df_places, df_veh], ['usagers-2021', 'caracteristiques-2021', 'lieux-2021', 'vehicules-2021']):
            output_path1 = os.path.join(old_filepath, f'{filename}.csv')
            file.to_csv(output_path1, index=False)
        
        remove_new_csv(csv_files)

def preprocess_data(df_users, df_caract, df_places, df_veh):
        #--Creating new columns
    nb_victim = pd.crosstab(df_users.Num_Acc, "count").reset_index()
    nb_vehicules = pd.crosstab(df_veh.Num_Acc, "count").reset_index()
    df_users["year_acc"] = df_users["Num_Acc"].astype(str).apply(lambda x : x[:4]).astype(int)
    df_users["victim_age"] = df_users["year_acc"]-df_users["an_nais"]
    for i in df_users["victim_age"] :
        if (i>120)|(i<0):
            df_users["victim_age"].replace(i,np.nan)
    df_caract["hour"] = df_caract["hrmn"].astype(str).apply(lambda x : x[:-3])
    df_caract.drop(['hrmn', 'an'], inplace=True, axis=1)
    df_users.drop(['an_nais'], inplace=True, axis=1)

    #--Replacing names 
    df_users.grav.replace([1,2,3,4], [1,3,4,2], inplace = True)
    df_caract.rename({"agg" : "agg_"},  inplace = True, axis = 1)
    df_caract.rename({"int" : "int_"},  inplace = True, axis = 1)
    corse_replace = {"2A":"201", "2B":"202"}
    df_caract["dep"] = df_caract["dep"].str.replace("2A", "201")
    df_caract["dep"] = df_caract["dep"].str.replace("2B", "202")
    df_caract["com"] = df_caract["com"].str.replace("2A", "201")
    df_caract["com"] = df_caract["com"].str.replace("2B", "202")

    #--Converting columns types
    df_caract[["dep","com", "hour"]] = df_caract[["dep","com", "hour"]].astype(int)

    dico_to_float = { 'lat': float, 'long':float}
    df_caract["lat"] = df_caract["lat"].str.replace(',', '.')
    df_caract["long"] = df_caract["long"].str.replace(',', '.')
    df_caract = df_caract.astype(dico_to_float)


    #--Grouping modalities 
    dico = {1:0, 2:1, 3:1, 4:1, 5:1, 6:1,7:1, 8:0, 9:0}
    df_caract["atm"] = df_caract["atm"].replace(dico)
    catv_value = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,30,31,32,33,34,35,36,37,38,39,40,41,42,43,50,60,80,99]
    catv_value_new = [0,1,1,2,1,1,6,2,5,5,5,5,5,4,4,4,4,4,3,3,4,4,1,1,1,1,1,6,6,3,3,3,3,1,1,1,1,1,0,0]
    df_veh['catv'].replace(catv_value, catv_value_new, inplace = True)

    #--Merging datasets 
    fusion1= df_users.merge(df_veh, on = ["Num_Acc","num_veh", "id_vehicule"], how="inner")
    fusion1 = fusion1.sort_values(by = "grav", ascending = False)
    fusion1 = fusion1.drop_duplicates(subset = ['Num_Acc'], keep="first")
    fusion2 = fusion1.merge(df_places, on = "Num_Acc", how = "left")
    df = fusion2.merge(df_caract, on = 'Num_Acc', how="left")

    #--Adding new columns
    df = df.merge(nb_victim, on = "Num_Acc", how = "inner")
    df.rename({"count" :"nb_victim"},axis = 1, inplace = True) 
    df = df.merge(nb_vehicules, on = "Num_Acc", how = "inner") 
    df.rename({"count" :"nb_vehicules"},axis = 1, inplace = True)

    #--Modification of the target variable  : 1 : prioritary // 0 : non-prioritary
    df['grav'].replace([2,3,4], [0,1,1], inplace=True)


    #--Replacing values -1 and 0 
    col_to_replace0_na = [ "trajet", "catv", "motor"]
    col_to_replace1_na = [ "trajet", "secu1", "catv", "obsm", "motor", "circ", "surf", "situ", "vma", "atm", "col"]
    df[col_to_replace1_na] = df[col_to_replace1_na].replace(-1, np.nan)
    df[col_to_replace0_na] = df[col_to_replace0_na].replace(0, np.nan)


    #--Dropping columns 
    list_to_drop = ['senc','larrout','actp', 'manv', 'choc', 'nbv', 'prof', 'plan', 'Num_Acc', 'id_vehicule', 'num_veh', 'pr', 'pr1','voie', 'trajet',"secu2", "secu3",'adr', 'v1', 'lartpc','occutc','v2','vosp','locp','etatp', 'infra', 'obs' ]
    df.drop(list_to_drop, axis=1, inplace=True)

    #--Dropping lines with NaN values
    col_to_drop_lines = ['catv', 'vma', 'secu1', 'obsm', 'atm']
    df = df.dropna(subset = col_to_drop_lines, axis=0)

    #--Saving the dataframes to their respective output file paths
    for file, filename in zip([df], ['current_data']):
        output_file = os.path.join(processed_folder, f'{filename}.csv')
        file.to_csv(output_file, index=False)

def preprocess_new_data():

    df_users = None
    df_caract = None
    df_places = None
    df_veh = None
    csv_files = check_new_csv(output_filepath)
    print("print all csv files {}".format(csv_files))
    if csv_files:
        for csv_file in csv_files:
            tmp_file = csv_file
            print(tmp_file)
            if csv_file[0:3] == 'usa':
                df_users = pd.read_csv(tmp_file, sep=";", on_bad_lines='skip')
                print(df_users)
            elif csv_file[0:3] == 'car':
                df_caract = pd.read_csv(tmp_file, sep=";", header=0, low_memory=False, on_bad_lines='skip')
            elif csv_file[0:3] == 'lie':
                df_places = pd.read_csv(tmp_file, sep=";", encoding='utf-8', on_bad_lines='skip')
            elif csv_file[0:3] == 'veh':
                df_veh = pd.read_csv(tmp_file, sep=";", on_bad_lines='skip')
        preprocess_data(df_users, df_caract, df_places, df_veh)

def find_latest_versioned_model(base_filename):
    """
    Find the latest versioned model file based on base_filename.
    Returns the path to the latest versioned model file.
    """
    search_pattern = f"{base_filename}-v*-*.joblib"
    files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No model files found with pattern '{search_pattern}'")
    
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def get_prediction():
    """
    Generates predictions for reference and current data using RandomForestClassifier.
    """
    # Create a copy of the dataframes to avoid modifying the original data
    # Load train and test datasets
    X_train = pd.read_csv(f'{processed_folder}X_train.csv')
    y_train = pd.read_csv(f'{processed_folder}y_train.csv')
    reference_data = X_train.copy()
    reference_data['target'] = y_train
    y_train = np.ravel(y_train)

    # base_model_filename = Config.TRAINED_MODEL_DIR
    # Find the latest versioned model file
    latest_model_file = find_latest_versioned_model(base_joblib_filename)
    # Load the model
    model = joblib.load(latest_model_file)

    # Generate predictions for reference and current data
    reference_data['prediction'] = model.predict_proba(X_train)[:, 1]

    df = pd.read_csv(f'{processed_folder}current_data.csv')
    targets = df['grav']
    X_test = df.drop(['grav'], axis = 1)
    current_data = X_test.copy()
    current_data['targets'] = targets
    y_test = np.ravel(targets)
    current_data['prediction'] = model.predict_proba(X_test)[:, 1]
    return reference_data, current_data

def generate_classification_report():
    """
    Generates a classification report using Evidently.
    """
    reference_data, current_data = get_prediction()
    # TODO : Create a Report instance for classification with a set of predefined metrics.
    # Use the ClassificationPreset with probas_threshold=0.5
    classification_report = Report(metrics=[
        ClassificationPreset(probas_threshold=0.5),
    ])

    # Generate the report
    classification_report.run(reference_data=reference_data, current_data=current_data)

    test = classification_report.as_dict()
    f1_current = test['metrics'][0]['result']['current']['f1']
    f1_reference = test['metrics'][0]['result']['reference']['f1']
    f1_diff =  f1_reference - f1_current

    if f1_diff > 0:
        print('success')
    else:
        raise Exception('the current model works perfect')
    
def retrain_model():
    merge_old_new()
    compose_file = "../../docker/docker-compose.model.yml"
    subprocess.run(["docker-compose", "-f", compose_file, "up", "-d"], check=True)


with DAG(
    dag_id='new_data_ingestion',
    tags=['new_data_ingestion', 'datascientest','MLops'],
    schedule_interval=None,
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False
) as dag:

    task1 = PythonOperator(
        task_id = 'task1_generate_new_data',
        python_callable = generate_new_data
    )

    task2 = PythonOperator(
        task_id = 'task2_preprocess_new_data',
        python_callable = preprocess_new_data
    )

    task3 = PythonOperator(
        task_id = 'task3_generate_classification_report',
        python_callable = generate_classification_report
    )

    task4 = PythonOperator(
        task_id = 'task4_retrain_model',
        python_callable = retrain_model,
        trigger_rule='all_success'
    )


    task1 >> task2 >> task3 >> task4