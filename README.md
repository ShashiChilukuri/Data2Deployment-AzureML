# Data to Model deployment of Heart Failure Prediction in Azure ML

This is a capstone project as part of the Udacity Azure ML Nanodegree program. The aim of this project is the perform all the tasks from picking the dataset to deployment of the model in the Azure Machine Learning. As part of this project, I have used a Heart Failure Prediction dataset ([from Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)) to build the prediciton classifier. The primary goal of this binary classifier is to predict the mortality casused by Heart Failure based on 12 features. To perform this task with best model, two types of training methods were done namely - Azure AutoML and HyperDrive. From each of these methods, a model is created. The model which perfromed best is selected for deployment and later consumed its REST endpoint. Below is the project workflow diagram:

![workflow_diagam](/Users/shashi/Documents/Job/Azure/Capstone-MLE_with_Azure/Data2Deployment-AzureML/Assets/workflow_diagram.png)

 

## Project Set Up and Installation
Project requirements:

* Jupyter Notebook
* Python 3.8
* Azure ML

To perform this project tasks, requires to setup azure ML studio workspace, compute instance/cluster and Azure SDK.

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
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

The best model found using AutoML method is Voting Ensemble, which is based on 7 different ensemble models each with specific weightage as shown below:



![image-20220627205552309](/Users/shashi/Library/Application Support/typora-user-images/image-20220627205552309.png)

![image-20220627205633532](/Users/shashi/Library/Application Support/typora-user-images/image-20220627205633532.png)



![image-20220627210826530](/Users/shashi/Library/Application Support/typora-user-images/image-20220627210826530.png)

![image-20220627211250856](/Users/shashi/Library/Application Support/typora-user-images/image-20220627211250856.png)

![image-20220627211328967](/Users/shashi/Library/Application Support/typora-user-images/image-20220627211328967.png)

![image-20220627211514482](/Users/shashi/Library/Application Support/typora-user-images/image-20220627211514482.png)

![image-20220627211710453](/Users/shashi/Library/Application Support/typora-user-images/image-20220627211710453.png)



Some areas of improvement for future AutoML experiments are:

- Need to try different sampling methods such as Grid search etc. need to be tested (as it is more comprehensive compared to Random Search)
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

The best model run accuracy found using HyperDrive method is 75% as shown below:

![image-20220627212025064](/Users/shashi/Library/Application Support/typora-user-images/image-20220627212025064.png)



![image-20220627212539331](/Users/shashi/Library/Application Support/typora-user-images/image-20220627212539331.png)

![image-20220627212616130](/Users/shashi/Library/Application Support/typora-user-images/image-20220627212616130.png)













Some areas of improvement for future experiments with HyperDrive is:

- Need to try with different classification models with HyperDrive. For this experiment, used Random Forest mode, In the feature, would like to test a statistical model and also other ensumble models like XGBoost.  
- Need to try with more wider range of hyper parameters in the HyperDrive

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

![image-20220627212941285](/Users/shashi/Library/Application Support/typora-user-images/image-20220627212941285.png)

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

![image-20220627213303344](/Users/shashi/Library/Application Support/typora-user-images/image-20220627213303344.png)

Deployment was successful, we can see following screenshots to confirm that:

![image-20220627213651720](/Users/shashi/Library/Application Support/typora-user-images/image-20220627213651720.png)

![image-20220627213733073](/Users/shashi/Library/Application Support/typora-user-images/image-20220627213733073.png)

To test the deployed model, following two methods where used:

* REST endpoint was used in the endpoint.py script to test from command line. 

  ![image-20220627214719660](/Users/shashi/Library/Application Support/typora-user-images/image-20220627214719660.png)

* Request.post method was used in the notebook with Scoring_uri and test data to get the response.

  ![image-20220627214549261](/Users/shashi/Library/Application Support/typora-user-images/image-20220627214549261.png)

## Screen Recording
Screencast

https://www.youtube.com/watch?v=Lzl7Y3H7leo

*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

