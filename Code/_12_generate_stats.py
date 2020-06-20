import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from rfpimp import permutation_importances

from _11_stats import plot_confusion_matrix, pandas_classification_report, \
                      imp_df, drop_col_feat_imp, accuracy

np_load_old = np.load
# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

def generate_stats(simulation_folder, featured_model="RF", test_version=''):
    """
    Function for generating statistics about model
    :param scenario: "Old" or other if defined
    :param featured_model: "RF", "GB" or other id defined
    :return: statistics
    """

    Start = datetime.now()
    project_directory = os.path.dirname(os.getcwd())
    path_to_data = os.path.join(project_directory, "Data", "Synthetic data")
    path_to_characteristics_data = os.path.join(path_to_data, simulation_folder,
                                                "Characteristics"+test_version)
    path_to_scenario = os.path.join(project_directory, "Models", featured_model,
                                    simulation_folder, "Model"+test_version)
    path_to_stats = os.path.join(path_to_scenario, "Stats")  #"C:\\Hania\\Praca\\TeX\\Notatki-klasyfikatory" #
    if not os.path.exists(path_to_stats):
        os.makedirs(path_to_stats)
    path_to_model = os.path.join(path_to_scenario, "model.sav")
    path_to_labelencoder = os.path.join(path_to_characteristics_data, 'classes.npy')
    labelencoder = LabelEncoder()
    classes = [x.split("_")[0] for x in np.load(path_to_labelencoder)]
    labelencoder.classes_ = classes
    X_train = np.load(os.path.join(path_to_characteristics_data, "X_train.npy"))
    X_test = np.load(os.path.join(path_to_characteristics_data, "X_test.npy"))
    y_train = np.load(os.path.join(path_to_characteristics_data, "y_train.npy"))
    y_test = np.load(os.path.join(path_to_characteristics_data, "y_test.npy"))
    # TODO: fix the save of the data to get variable names from there
    characteristics_data = pd.read_csv(os.path.join(path_to_characteristics_data, "characteristics.csv"))
    model = joblib.load(path_to_model)
    data_type = ["Train", "Test"]
    for dt in data_type:
        X = X_train if dt == "Train" else X_test
        y = y_train if dt == "Train" else y_test
        # Making the Confusion Matrix
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        # Plot non-normalized confusion matrix
        fig = plt.figure()
        plot_confusion_matrix(cm, classes=labelencoder.classes_, title='Confusion matrix, without normalization')
        fig.savefig(os.path.join(path_to_stats, "Confusion_Matrix_NotNormalized_" + dt + ".pdf"), dpi=fig.dpi)
        plt.close()

#        # Plot normalized confusion matrix
#        fig = plt.figure()
#        vers = "no D" if "_noD" in test_version else "with D"
#        ts = "_Three_noD_" if "_noD" in test_version else "_Three_"
#        plot_confusion_matrix(cm, classes=labelencoder.classes_, normalize=True, title=featured_model+", "+vers)
#        fig.savefig(os.path.join(path_to_stats, featured_model+ts+"Confusion_Matrix_Normalized_" + dt + ".pdf"), dpi=fig.dpi)
#        plt.close()
        
        # Plot normalized confusion matrix
        fig = plt.figure()
        vers = "no D" if "_noD" in test_version else "with D"
        plot_confusion_matrix(cm, classes=labelencoder.classes_, normalize=True, title=featured_model+", "+vers)
        fig.savefig(os.path.join(path_to_stats, "Confusion_Matrix_Normalized_" + dt + ".pdf"), dpi=fig.dpi)
        plt.close()

        # class report
        print("class report")
        report = classification_report(y, y_pred, target_names=labelencoder.classes_, digits=3)
        report = pandas_classification_report(report)
        report.to_csv(os.path.join(path_to_stats, "Classification_Report_" + dt + ".csv"))

        # accuracy
        print("acc")
        acu = accuracy_score(y, y_pred)
        df = pd.DataFrame({'acc': [acu]})
        df.to_csv(os.path.join(path_to_stats, "Accuracy_" + dt + ".csv"))
        

    # feature importances
    importances = model.feature_importances_  
    if "_noD" in test_version:
        column_names = characteristics_data.drop(["file", "motion", "diff_type", "D"], axis=1).columns.values
    else:
        column_names = characteristics_data.drop(["file", "motion", "diff_type"], axis=1).columns.values
    df = imp_df(column_names, importances)
    df.to_csv(os.path.join(path_to_stats, "Feature_importances.csv"), index=False)
    
    
#    # permutation importances
#    X_train_df = pd.DataFrame(X_train, columns=column_names)
#    y_train_df = pd.DataFrame(y_train)
#    df = permutation_importances(clone(model), X_train_df, y_train_df, accuracy)
#    df.to_csv(os.path.join(path_to_stats, "Permutation_fi.csv"), index=True)
#    
#    # drop column feature importance
#    # FIX: after change X_train etc. to contain data names, change the snippet here
#    X_train_df = pd.DataFrame(X_train, columns=column_names)
#    df = drop_col_feat_imp(model, X_train_df, y_train)
#    df.to_csv(os.path.join(path_to_stats, "Drop_column_fi.csv"), index=False)

    End = datetime.now()
    ExecutedTime = End - Start
    df = pd.DataFrame({'ExecutedTime': [ExecutedTime]})
    df.to_csv(os.path.join(path_to_stats, "time_for_stats_generator.csv"))
    print(ExecutedTime)


if __name__ == "__main__":
    
    # TODO: no drop column
    generate_stats(simulation_folder="Base_corr", featured_model='RF', test_version='_sta_10')
    generate_stats(simulation_folder="Base_corr", featured_model='GB', test_version='_sta_10')
    generate_stats(simulation_folder="Base_corr", featured_model='RF', test_version='_sta_10_noD')
    generate_stats(simulation_folder="Base_corr", featured_model='GB', test_version='_sta_10_noD')
    
    # restore np.load for future normal usage
    np.load = np_load_old
