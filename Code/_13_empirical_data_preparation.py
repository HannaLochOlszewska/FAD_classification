import os
import numpy as np
import pandas as pd
import joblib

#from sklearn.preprocessing import LabelEncoder

from _05_characteristics import Characteristic, CharacteristicTwo, CharacteristicThree
from _06_generate_characteristics import CharSet

def prepare_data():
    ## TODO: PASTE THE CODE FOR GENERATION OF .NPY FILE!
    pass

def calculate_characteristics(file, data_folder, set_name, dt, char_set):
    
    project_directory = os.path.dirname(os.getcwd())
    path_to_data = os.path.join(project_directory, "Data", "Empirical data", data_folder)
    trajectories = np.load(os.path.join(path_to_data, "Trajectories", file), allow_pickle=True)
        
    characteristics_full = pd.DataFrame([])
    for trajectory in trajectories:
        #TODO: sprawdz indeksowanie
        name = trajectory[0]
        print(name)
        X = trajectory[1][:,0]
        Y = trajectory[1][:,1]
        
        if char_set == CharSet.One:
            ch = Characteristic(x=X, y=Y, dt=dt, percentage_max_n=1, typ='', motion='', file=name)
        elif char_set == CharSet.Two:
            ch = CharacteristicTwo(x=X, y=Y, dt=dt, percentage_max_n=0.1, typ='', motion='', file=name)
        elif char_set == CharSet.Three:
            ch = CharacteristicThree(x=X, y=Y, dt=dt, percentage_max_n=0.1, typ='', motion='', file=name)

        data = ch.data
        characteristics_full = pd.concat([characteristics_full, data], sort=False)
        
    save_dir = os.path.join(path_to_data, "Characteristics"+set_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    characteristics_full.to_csv(os.path.join(save_dir, "characteristics.csv"), index=False)


def prepare_input(data_folder, set_name):
    
    project_directory = os.path.dirname(os.getcwd())
    path_to_data = os.path.join(project_directory, "Data", "Empirical data", data_folder)
    path_to_characteristics_data = os.path.join(path_to_data, "Characteristics"+set_name)
    file_with_characteristics = os.path.join(path_to_characteristics_data, "characteristics.csv")
    characteristics_data = pd.read_csv(file_with_characteristics)
    if "_noD" in set_name:
        characteristics_data = characteristics_data.drop(["diff_type", "motion", "D"], axis=1)
    else:
        characteristics_data = characteristics_data.drop(["diff_type", "motion"], axis=1)
    
    X = characteristics_data.iloc[:,1:].values
    y = characteristics_data["file"].values
    
    np.save(os.path.join(path_to_characteristics_data, "X_data.npy"), X)
    np.save(os.path.join(path_to_characteristics_data, "filenames_data.npy"), y)
    
    return


def calculate_prognosis(data_folder, test_version_2, set_name, simulation_folder, featured_model):

    project_directory = os.path.dirname(os.getcwd())
    path_to_data = os.path.join(project_directory, "Data", "Empirical data", data_folder)
    path_to_characteristics_data = os.path.join(path_to_data, "Characteristics"+set_name)
    
    path_to_model = os.path.join(project_directory, "Models", featured_model, 
                                 simulation_folder, "Model"+test_version_2)
    path_to_save = os.path.join(path_to_model, 'Empirical', data_folder)
        
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    
    model = joblib.load(os.path.join(path_to_model, "model.sav"))
              
    X_data = np.load(os.path.join(path_to_characteristics_data, "X_data.npy"), allow_pickle=True)
    y_pred = list(model.predict(X_data))
    
    
    path_to_labelencoder = os.path.join(project_directory, "Data", "Synthetic data",
                                        simulation_folder, "Characteristics"+test_version_2, 'classes.npy')
        
#    labelencoder = LabelEncoder()
#    labelencoder.classes_ = np.load(path_to_labelencoder)
    classes = np.load(path_to_labelencoder, allow_pickle=True)
    
    results = pd.DataFrame([])
    columns = ["diff_type", "no_paths", "percentage"]
    for i in range(len(classes)):
        data = pd.DataFrame([[classes[i], y_pred.count(i), y_pred.count(i)/len(y_pred)]], 
                             columns=columns)
        results = pd.concat([results, data], sort=False)
    results.to_csv(os.path.join(path_to_save, "results"+set_name+".csv"), index=False)
    
    return

    
# TODO:
# 1) załadować enkoder dla modelu - sciezka?

dt = 2.84 #in sec.

data_folder = "Beethoven_AWJJEB"
simulation_folder = "Zero_free"

test_version_2 = "_sta_10"  # base_version
char_set = CharSet.Three

file = "a2AR_basal_TC685_names.npy"
prefix = "_a2AR_basal"
set_name = prefix + test_version_2 # test_version
calculate_characteristics(file, data_folder, set_name, dt, char_set)
prepare_input(data_folder, set_name)
calculate_prognosis(data_folder, test_version_2, set_name, simulation_folder, featured_model='RF')
calculate_prognosis(data_folder, test_version_2, set_name, simulation_folder, featured_model='GB')


file = "Gi_basal_TC685_names.npy"
prefix = "_Gi_basal"
set_name = prefix + test_version_2 # test_version
calculate_characteristics(file, data_folder, set_name, dt, char_set)
prepare_input(data_folder, set_name)  
calculate_prognosis(data_folder, test_version_2, set_name, simulation_folder, featured_model='RF')
calculate_prognosis(data_folder, test_version_2, set_name, simulation_folder, featured_model='GB')
    
# TODO: zmienic test_version_2 na cos innego
# TODO: uporzadkowac kolejnosc zmiennych w funkcji