import os
import joblib
import argparse
import numpy as np
import pandas as pd


from azureml.core.run import Run
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from azureml.data.dataset_factory import TabularDatasetFactory

# Creating tabular data from the data path
data_path = "https://github.com/ShashiChilukuri/Data2Deployment-AzureML/blob/bc2160defa96caf4a11b509e1d7ce59a47bab792/heart_failure_clinical_records_dataset.csv"
data = TabularDatasetFactory.from_delimited_files(data_path)

#Saving model for current iteration
run = Run.get_context()

# Cleaning and splitting features and label data
x_df = data.to_pandas_dataframe().dropna()
y_df = x_df.pop("DEATH_EVENT")

# Splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

def main():
    # Adding arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=20, help="N.O trees in the forest")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Min samples to split")
    
    args = parser.parse_args(args=[])
    
    run.log("N.O trees in the forest:", np.int(args.n_estimators))
    run.log("Min samples to split:", np.int(args.min_samples_split))
    
    model = RandomForestClassifier(n_estimators=args.n_estimators, 
                                   random_state=22, 
                                   min_samples_split=args.min_samples_split).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", float(accuracy))

    # Saving model for current iteration
    os.makedirs('outputs', exist_ok=True)
    # joblib.dump(model, 'outputs/hyperDrive_{}_{}'.format(args.n_estimators, args.min_samples_split))
    joblib.dump(value=model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()


