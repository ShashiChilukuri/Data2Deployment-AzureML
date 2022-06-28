# Data to Model deployment of Heart Failure Prediction in Azure ML

This is a capstone project as part of the Udacity Azure ML Nanodegree program. The aim of this project is the perform all the tasks from picking the dataset to deployment of the model in the Azure Machine Learning. As part of this project, I have used a Heart Failure Prediction dataset ([from Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)) to build the prediciton classifier. The primary goal of this binary classifier is to predict the mortality casused by Heart Failure based on 12 features. To perform this task with best model, two types of training methods were done namely - Azure AutoML and HyperDrive. From each of these methods, a model is created. The model which perfromed best is selected for deployment and later consumed its REST endpoint. Below is the project workflow diagram (Source: udacity):

![workflow_diagam](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/workflow_diagram.png)

 

## Project Set Up and Installation
Project requirements:

* Jupyter Notebook
* Python 3.8
* Azure ML

Note: To perform this project tasks, requires to setup azure ML studio workspace, compute instance/cluster and Azure SDK.

## Dataset

### Overview
The datset used for this project is a Heart Failure Prediction dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data).  This dataset containes 12 features and a label to classify death by heart failure. 

### Task
The primary task of this project is to predict the mortality casused by Heart Failure based on 12 features. here is the list of features and label and its description:

1. age: Age of a person
2. anaemia: Decrease of red blood cells of hemoglobin (boolean)
3. creatinine_phosphokinase: Level of the CPK enzyme in the blood (mcg/L)
4. diabetes: If the patient has diabetes (boolean)
5. ejection_fraction: Percentage of blood leaving the heart at each contraction (percentage)
6. high_blood_pressure: If the patient has hypertension (boolean)
7. platelets: Platelets in the blood (kiloplatelets/mL)
8. serum_creatinine: Level of serum sodium in the blood (mEq/L)
9. serum_sodium: Level of serum sodium in the blood (mEq/L)
10. Time: follow up period(days)
11. Smoking: True-1, False-0
12. sex: Woman or man (binary)

*DEATH_EVENT*: Label (True- 1, False-0) 

### Access

This is a publicly avaiable Kaggle dataset. As part of the project, uploaded this dataset to this project repository for it use.

## Automated ML
AutoML (Automated Machine Learning) essentially automates all accepts of machine learning process i.e, feature engineering, selection of hyperparameters, model training etc.

AutoML config class is used for submittting an automated ML experiment in the Azure Machine learning. Auto ML settings helps to moderate how we want our experiment to be run. In this case, wanted to experiment to timeout in 30 minutes, with max iterations to be executed in parallel is 5 and cross validation to perform is 2. Although there are many metrics, I choose to pick "accuracy" metric, as it would be good metric for simple datasets. These automl settings are passed on to AutoMLConfig class along with the compute instance, data, task type and label.

```python
# Automl settings
automl_settings = {"experiment_timeout_minutes": 30,
                   "max_concurrent_iterations": 5,
                   "n_cross_validations": 2,
                   "primary_metric": 'accuracy',
                   "verbosity": logging.INFO
                  }

# Defining automl config
automl_config = AutoMLConfig(task='classification',
                             compute_target=compute_target,
                             training_data=data,
                             label_column_name='DEATH_EVENT',
                             **automl_settings)
```

### Results
The best model found using AutoML method is Voting Ensemble with an accuracy of 86.2%, which is based on 7 different ensemble models each with specific weightage.

Here is the screen shot of model experiment submitted.

![image-2](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/automl1.png)

![image-3](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/automl2.png)

Here is the screenshot of AutoML job was running and when it is completed:

![image-4](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/automl3.png)

![image-5](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/automl4.png)

Here is the best model with accuracy:

![image-6](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/automl5.png)

Here is the snapshot of run details:

![image-7](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/automl6.png)

Here are the parameters for the AutoML model:

```Python
"properties": {
        "num_iterations": "1000",
        "training_type": "TrainFull",
        "acquisition_function": "EI",
        "primary_metric": "accuracy",
        "train_split": "0",
        "acquisition_parameter": "0",
        "num_cross_validation": "2",
        "target": "Compute-Standard",
        "AMLSettingsJsonString": '{"path":null,"name":"ClassifyHeartFailure-AutoML","subscription_id":"6971f5ac-8af1-446e-8034-05acea24681f","resource_group":"aml-quickstarts-199609","workspace_name":"quick-starts-ws-199609","region":"southcentralus","compute_target":"Compute-Standard","spark_service":null,"azure_service":"remote","many_models":false,"pipeline_fetch_max_batch_size":1,"enable_batch_run":true,"enable_run_restructure":false,"start_auxiliary_runs_before_parent_complete":false,"enable_code_generation":true,"iterations":1000,"primary_metric":"accuracy","task_type":"classification","positive_label":null,"data_script":null,"test_size":0.0,"test_include_predictions_only":false,"validation_size":0.0,"n_cross_validations":2,"y_min":null,"y_max":null,"num_classes":null,"featurization":"auto","_ignore_package_version_incompatibilities":false,"is_timeseries":false,"max_cores_per_iteration":1,"max_concurrent_iterations":5,"iteration_timeout_minutes":null,"mem_in_mb":null,"enforce_time_on_windows":false,"experiment_timeout_minutes":30,"experiment_exit_score":null,"partition_column_names":null,"whitelist_models":null,"blacklist_algos":["TensorFlowLinearClassifier","TensorFlowDNN"],"supported_models":["SGD","XGBoostClassifier","AveragedPerceptronClassifier","BernoulliNaiveBayes","KNN","DecisionTree","LogisticRegression","TensorFlowLinearClassifier","ExtremeRandomTrees","MultinomialNaiveBayes","RandomForest","TabnetClassifier","TensorFlowDNN","LinearSVM","GradientBoosting","SVM","LightGBM"],"private_models":[],"auto_blacklist":true,"blacklist_samples_reached":false,"exclude_nan_labels":true,"verbosity":20,"_debug_log":"azureml_automl.log","show_warnings":false,"model_explainability":true,"service_url":null,"sdk_url":null,"sdk_packages":null,"enable_onnx_compatible_models":false,"enable_split_onnx_featurizer_estimator_models":false,"vm_type":"STANDARD_DS3_V2","telemetry_verbosity":20,"send_telemetry":true,"enable_dnn":false,"scenario":"SDK-1.13.0","environment_label":null,"save_mlflow":false,"enable_categorical_indicators":false,"force_text_dnn":false,"enable_feature_sweeping":true,"enable_early_stopping":true,"early_stopping_n_iters":10,"arguments":null,"dataset_id":"3689506b-c445-4bd7-a527-46695ffe5ece","hyperdrive_config":null,"validation_dataset_id":null,"run_source":null,"metrics":null,"enable_metric_confidence":false,"enable_ensembling":true,"enable_stack_ensembling":true,"ensemble_iterations":15,"enable_tf":false,"enable_subsampling":null,"subsample_seed":null,"enable_nimbusml":false,"enable_streaming":false,"force_streaming":false,"track_child_runs":true,"allowed_private_models":[],"label_column_name":"DEATH_EVENT","weight_column_name":null,"cv_split_column_names":null,"enable_local_managed":false,"_local_managed_run_id":null,"cost_mode":1,"lag_length":0,"metric_operation":"maximize","preprocess":true}',
        "DataPrepJsonString": '{\\"training_data\\": {\\"datasetId\\": \\"3689506b-c445-4bd7-a527-46695ffe5ece\\"}, \\"datasets\\": 0}',
        "EnableSubsampling": None,
        "runTemplate": "AutoML",
        "azureml.runsource": "automl",
        "display_task_type": "classification",
        "dependencies_versions": '{"azureml-widgets": "1.42.0", "azureml-training-tabular": "1.42.0", "azureml-train": "1.42.0", "azureml-train-restclients-hyperdrive": "1.42.0", "azureml-train-core": "1.42.0", "azureml-train-automl": "1.42.0", "azureml-train-automl-runtime": "1.42.0", "azureml-train-automl-client": "1.42.0", "azureml-tensorboard": "1.42.0", "azureml-telemetry": "1.42.0", "azureml-sdk": "1.42.0", "azureml-samples": "0 unknown", "azureml-responsibleai": "1.42.0", "azureml-pipeline": "1.42.0", "azureml-pipeline-steps": "1.42.0", "azureml-pipeline-core": "1.42.0", "azureml-opendatasets": "1.42.0", "azureml-mlflow": "1.42.0", "azureml-interpret": "1.42.0", "azureml-inference-server-http": "0.4.13", "azureml-explain-model": "1.42.0", "azureml-defaults": "1.42.0", "azureml-dataset-runtime": "1.42.0", "azureml-dataprep": "4.0.1", "azureml-dataprep-rslex": "2.6.1", "azureml-dataprep-native": "38.0.0", "azureml-datadrift": "1.42.0", "azureml-core": "1.42.0", "azureml-contrib-services": "1.42.0", "azureml-contrib-server": "1.42.0", "azureml-contrib-reinforcementlearning": "1.42.0", "azureml-contrib-pipeline-steps": "1.42.0", "azureml-contrib-notebook": "1.42.0", "azureml-contrib-fairness": "1.42.0", "azureml-contrib-dataset": "1.42.0", "azureml-contrib-automl-pipeline-steps": "1.42.0", "azureml-cli-common": "1.42.0", "azureml-automl-runtime": "1.42.0", "azureml-automl-dnn-nlp": "1.42.0", "azureml-automl-core": "1.42.0", "azureml-accel-models": "1.42.0"}',
        "_aml_system_scenario_identification": "Remote.Parent",
        "ClientType": "SDK",
        "environment_cpu_name": "AzureML-AutoML",
        "environment_cpu_label": "prod",
        "environment_gpu_name": "AzureML-AutoML-GPU",
        "environment_gpu_label": "prod",
        "root_attribution": "automl",
        "attribution": "AutoML",
        "Orchestrator": "AutoML",
        "CancelUri": "https://southcentralus.api.azureml.ms/jasmine/v1.0/subscriptions/6971f5ac-8af1-446e-8034-05acea24681f/resourceGroups/aml-quickstarts-199609/providers/Microsoft.MachineLearningServices/workspaces/quick-starts-ws-199609/experimentids/6f19f0b0-3cd6-42fd-aa9f-d8152ae5cdf2/cancel/AutoML_580b1e1a-0ea0-4980-9412-3c88532c0271",
        "ClientSdkVersion": "1.42.0.post1",
        "snapshotId": "00000000-0000-0000-0000-000000000000",
        "SetupRunId": "AutoML_580b1e1a-0ea0-4980-9412-3c88532c0271_setup",
        "SetupRunContainerId": "dcid.AutoML_580b1e1a-0ea0-4980-9412-3c88532c0271_setup",
        "FeaturizationRunJsonPath": "featurizer_container.json",
        "FeaturizationRunId": "AutoML_580b1e1a-0ea0-4980-9412-3c88532c0271_featurize",
        "ProblemInfoJsonString": '{"dataset_num_categorical": 0, "is_sparse": false, "subsampling": false, "has_extra_col": true, "dataset_classes": 2, "dataset_features": 12, "dataset_samples": 299, "single_frequency_class_detected": false}',
        "ModelExplainRunId": "AutoML_580b1e1a-0ea0-4980-9412-3c88532c0271_ModelExplain",
    }
```

![image-8](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/automl7.png)

Here is the best model pipeline with parameters:

![image-9](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/automl8.png)

Some areas of improvement for future AutoML experiments are:

- Need to try different sampling methods such as Grid search etc. (as it is more comprehensive compared to Random Search)
- Need to try different termination policies and compare how it performes.

## Hyperparameter Tuning
To predict heart failure, used Random Forest model and to fine tune the model parameters, used the Azure HyperDrive functionality. HyperDrive needs parameter sampler and early stopping policy to be feed in. For parameter sampling, used Random paramter sampling to sample over a hyperparameter search space. Picked this because this it is quicker than Grid search sampler as the parameter selection is random in nature. With respect to early stopping, I used Bandit early terminatin policy. Reason for selecting Bandit early termination policy is that it allows to select an interval and once it exceeds the specified interval, this policy will ends the job. It easy to use and provides more flexibility over other stopping policies such as median stopping.

Hyper Drive config setting guides in picking the best model. For this configuration, along with the parameter sampling and policy, used "accuracy" as primary metric as it is good metric for simple datasets, and the goal of this metric is to maximize as higher the accuracy better the model is. While the max total runs is 20 and concurrently it can run upto 4 runs.

```Python
# Early termination policy. This is not required if you are using Bayesian sampling.
early_termination_policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

#Create the different params that you will be using during training
parameter_space = {"--n_estimators": choice(10, 20, 40), "--min_samples_split": choice(2,4,6)}
param_sampling = RandomParameterSampling(parameter_space = parameter_space)

# Setup environment for your training run
sklearn_env = Environment.from_conda_specification(name='sklearn-env', file_path='conda_dependencies.yml')

# Create a ScriptRunConfig Object to specify the configuration details of your training job
src = ScriptRunConfig(source_directory = ".",
                      script='train.py',
                      compute_target=cluster_name,
                      environment = sklearn_env)

hyperdrive_run_config = HyperDriveConfig(run_config=src,
                                         hyperparameter_sampling=param_sampling,
                                         primary_metric_name='Accuracy',
                                         primary_metric_goal= PrimaryMetricGoal("MAXIMIZE"),
                                         max_total_runs=20,
                                         max_concurrent_runs=4,
                                         policy=early_termination_policy)
```

Reasons for picking the Random forest model is that it is a non-linear non-statistical tree classifier. As it is non-statistical model, there is no prior assumptions involved. And also, it requires little data preparation and robust against co-linearity & outliers.


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

The best model run accuracy found using HyperDrive method is 78.8%. Below is the screenshot of Run Details:

![image-10](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/hyperdrive1.png)

Here is the screenshot of  job completed successfully:

![image-11](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/hyperdrive2.png)

Below screenshot shows the best model accuracy and its parameters:

* Best model accuracy: 78.8%
* Best model Paramteters:
  * N.o trees in the forest: 20
  * Min Samples to split: 2

![image-12](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/hyperdrive3.png)

Some areas of improvement for future experiments with HyperDrive is:

- Need to try with different classification models with HyperDrive. For this experiment, used Random Forest mode, In the feature, would like to test a statistical model and also other ensumble models like XGBoost.  
- Need to try with more wider range of hyper parameters in the HyperDrive

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

![image-z](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/deploy2.png)

As part of the project, trained both AutoML model and also the Hyper drive based model. Best model picked out of these two methods is Voting Ensemble model using Auto ML method. It has an accuracy of 86.2%. This model is then deployed using Azure container Instance (ACI). To deploy the model, following code is used:

```Python
# Deploying the model
deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
service = Model.deploy(ws, 
                       "myservice", 
                       [best_automl_model], 
                       inference_config, 
                       deployment_config)

service.wait_for_deployment(show_output = True)
```

Here is the screenshot of model deployment in progress

![image-14](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/deploy3.png)

Deployment was successful, we can see following screenshots to confirm that:

![image-15](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/deploy4.png)

![image-16](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/deploy5.png)

To test the deployed model, following two methods where used:

1. REST endpoint was used in the endpoint.py script to test from command line as shown below. 

   ![image-17](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/test1.png)

2. Request.post method was used in the notebook with Scoring_uri and test data to get the response.

   ![image-17](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/test2.png)

## Screen Recording
Here is the link to the screencast: [E2E with Azure](https://www.youtube.com/watch?v=Lzl7Y3H7leo)

