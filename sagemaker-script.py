import argparse
import boto3
import time
import os
import io
import json
import sys
import pandas as pd
import numpy as np
import sagemaker.amazon.common as smac
from sagemaker.amazon.amazon_estimator import get_image_uri

# Connect to Dynamo for scanning our table and S3 for getting a CSV version of our data
s3 = boto3.client('s3')
sm = boto3.client('sagemaker')

fields = ["ID", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
          "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
          "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
          "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
          "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
          "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]


def train_model_deploy(args):

    backup_bucket = args.s3_backup_bucket
    sagemaker_bucket = args.s3_sagemaker_bucket
    role = args.role_arn
    sm_prefix = 'demo-breast-cancer-prediction'
    # Get Docker image for linear-learner
    container = get_image_uri(boto3.Session().region_name, 'linear-learner')

    # Find the latest item from the backup bucket
    objs = s3.list_objects(Bucket=backup_bucket)
    key_time = [(item['Key'], item['LastModified']) for item in objs['Contents']]
    key_time = sorted(key_time, key=lambda tup: tup[1], reverse=True)
    s3_file_key = key_time[0][0]

    print('Variables intialized as:')
    print(f'Backup Bucket {backup_bucket}')
    print(f'Backup File Key {s3_file_key}')
    print(f'Role ARN {role}')
    print(f'Sagemaker Bucket {sagemaker_bucket}')
    print(f'Sagemaker Prefix {sm_prefix}')
    print(f'Container {container}')

    try:
        data = load_backup_data(backup_bucket, s3_file_key)
        train_X, train_y, val_X, val_y, test_X, test_y = split_data(data)
        save_train_val_to_s3(sagemaker_bucket, sm_prefix, train_X, train_y, val_X, val_y)
        linear_job = create_training_job(container, sagemaker_bucket, sm_prefix, role)
        model_name = linear_job
        create_model(container, role, linear_job, model_name)
        linear_endpoint = create_or_update_endpoint(model_name)
        test_endpoint(linear_endpoint, test_X, test_y, train_X, train_y)
        print('Success')
    except Exception as e:
        print(e)
        sys.exit()


def load_backup_data(backup_bucket, s3_file_key):
    print(f'Loading Breast Cancer backup data from S3 {backup_bucket}/{s3_file_key}')

    # Load the data in the backup from S3 into a pandas DataFrame
    obj = s3.get_object(Bucket=backup_bucket, Key=s3_file_key)
    data = pd.read_csv(io.BytesIO(obj['Body'].read()))

    # print the shape of the data file
    print(f'Shape of backup data {data.shape}')
    return data


def split_data(data):
    print('Splitting our data into train and validation sets')

    '''
    Split our data (randomly) into train / validation / test approx 80% / 10% / 10%
    '''

    # Setting seed to 0 so we all get the same values
    np.random.seed(0)

    rand_split = np.random.rand(len(data))
    train_list = rand_split < 0.8
    val_list = (rand_split >= 0.8) & (rand_split < 0.9)
    test_list = rand_split >= 0.9

    data_train = data[train_list]
    data_val = data[val_list]
    data_test = data[test_list]

    # We are dropping the ID column and converting M (Malignant) -> 1, B (Benign) -> 0 (for binary classification)
    train_y = ((data_train.iloc[:, 1] == 'M') + 0).values
    train_X = data_train.iloc[:, 2:].values

    val_y = ((data_val.iloc[:, 1] == 'M') + 0).values
    val_X = data_val.iloc[:, 2:].values

    test_y = ((data_test.iloc[:, 1] == 'M') + 0).values
    test_X = data_test.iloc[:, 2:].values

    print(f'train size {len(train_X)}, validation size {len(val_X)}')

    return train_X, train_y, val_X, val_y, test_X, test_y


def save_train_val_to_s3(sagemaker_bucket, sm_prefix, train_X, train_y, val_X, val_y):

    print('Saving training and validation data to S3')

    '''
    Now, we'll convert the datasets to the recordIO-wrapped protobuf format used by the Amazon SageMaker
    algorithms, and then upload this data to S3
    '''

    # First we'll convert the training data set
    train_file = 'linear_train.data'

    f = io.BytesIO()
    smac.write_numpy_to_dense_tensor(f, train_X.astype('float32'), train_y.astype('float32'))
    f.seek(0)

    s3_train_data_loc = os.path.join(sm_prefix, 'train', train_file)
    print('Saving training data in RecordIO format to {}'.format(s3_train_data_loc))
    boto3.Session().resource('s3').Bucket(sagemaker_bucket).Object(s3_train_data_loc).upload_fileobj(f)

    # Now we'll convert the validation data set
    validation_file = 'linear_validation.data'

    f = io.BytesIO()
    smac.write_numpy_to_dense_tensor(f, val_X.astype('float32'), val_y.astype('float32'))
    f.seek(0)

    s3_validation_data_loc = os.path.join(sm_prefix, 'validation', validation_file)
    print('Saving validation data in RecordIO format to {}'.format(s3_validation_data_loc))
    boto3.Session().resource('s3').Bucket(sagemaker_bucket).Object(s3_validation_data_loc).upload_fileobj(f)


def create_training_job(container, sagemaker_bucket, sm_prefix, role):
    print('Creating training job')

    '''
    Train

    Now we can begin to specify our linear model. Amazon SageMaker's Linear Learner actually fits many models in
    parallel, each with slightly different hyperparameters, and then returns the one with the best fit.
    This functionality is automatically enabled. We can influence this using parameters like:

    * num_models - to increase to total number of models run. The specified parameters will always be one of those
      models, but the algorithm also chooses models with nearby parameter values in order to find a solution nearby
      that may be more optimal. In this case, we're going to use the max of 32.
    * loss - which controls how we penalize mistakes in our model estimates. For this case, let's use absolute loss
      as we haven't spent much time cleaning the data, and absolute loss will be less sensitive to outliers.
    * wd or l1 - which control regularization. Regularization can prevent model overfitting by preventing our estimates
      from becoming too finely tuned to the training data, which can actually hurt generalizability. In this case, we'll
      leave these parameters as their default "auto" though.
    '''

    # See 'Algorithms Provided by Amazon SageMaker: Common Parameters' in the SageMaker documentation for an
    # explanation of these values.
    # For this example, we are going to use a linear learning model to train a binary classifier (Benign / Malignant)
    linear_job = 'DEMO-linear-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    print(f'Using Sagemaker Algo container {container}')
    print("Job name is:", linear_job)

    linear_training_params = {
        'RoleArn': role,
        'TrainingJobName': linear_job,
        'AlgorithmSpecification': {
            'TrainingImage': container,
            'TrainingInputMode': 'File'
        },
        'ResourceConfig': {
            'InstanceCount': 1,
            'InstanceType': 'ml.c4.2xlarge',
            'VolumeSizeInGB': 10
        },
        'InputDataConfig': [
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': 's3://{}/{}/train/'.format(sagemaker_bucket, sm_prefix),
                        'S3DataDistributionType': 'ShardedByS3Key'
                    }
                },
                'CompressionType': 'None',
                'RecordWrapperType': 'None'
            },
            {
                'ChannelName': 'validation',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': 's3://{}/{}/validation/'.format(sagemaker_bucket, sm_prefix),
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'CompressionType': 'None',
                'RecordWrapperType': 'None'
            }

        ],
        'OutputDataConfig': {
            'S3OutputPath': 's3://{}/{}/'.format(sagemaker_bucket, sm_prefix)
        },
        'HyperParameters': {
            'feature_dim': '30',
            'mini_batch_size': '100',
            'predictor_type': 'regressor',
            'epochs': '10',
            'num_models': '32',
            'loss': 'absolute_loss'
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 60 * 60
        }
    }

    '''
    Now let's kick off our training job in SageMaker's distributed, managed training, using the parameters we just
    created.  Because training is managed, we don't have to wait for our job to finish to continue, but for this case,
    let's use boto3's 'training_job_completed_or_stopped' waiter so we can ensure that the job has been started.
    '''

    print(f'Creating Sagemaker Training Job {linear_job}')
    sm.create_training_job(**linear_training_params)

    status = sm.describe_training_job(TrainingJobName=linear_job)['TrainingJobStatus']
    print(f'Training job status: {status}')
    sm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=linear_job)
    if status == 'Failed':
        message = sm.describe_training_job(TrainingJobName=linear_job)['FailureReason']
        print('Training failed with the following error: {}'.format(message))
        raise Exception('Training job failed')

    return linear_job


def create_model(container, role, linear_job, model_name):
    print(f'Creating model from training {linear_job} with same name')

    '''
    Host

    Now that we've trained the linear algorithm on our data, let's setup a model which can later be hosted. We will:

    1. Point to the scoring container
    2. Point to the model.tar.gz that came from training
    3. Create the hosting model
    '''

    linear_hosting_container = {
        'Image': container,
        'ModelDataUrl': sm.describe_training_job(TrainingJobName=linear_job)['ModelArtifacts']['S3ModelArtifacts']
    }

    create_model_response = sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer=linear_hosting_container)

    print(f'Model ARN: {create_model_response["ModelArn"]}')


def create_or_update_endpoint(model_name):
    print('Creating endpoint for our ML Model')

    '''
    Once we've setup a model, we can configure what our hosting endpoints should be. Here we specify:

    1. EC2 instance type to use for hosting
    2. Initial number of instances
    3. Our hosting model name
    '''

    linear_endpoint_config = 'DEMO-linear-endpoint-config-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    print(linear_endpoint_config)
    create_endpoint_config_response = sm.create_endpoint_config(
        EndpointConfigName=linear_endpoint_config,
        ProductionVariants=[{
            'InstanceType': 'ml.m4.xlarge',
            'InitialInstanceCount': 1,
            'ModelName': model_name,
            'VariantName': 'AllTraffic'}])

    print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])

    '''
    Now that we've specified how our endpoint should be configured, we can create them. This can be done in the
    background, but for now let's run a loop that updates us on the status of the endpoints so that we know when
    they are ready for use.
    '''

    linear_endpoint = 'DEMO-linear-endpoint-breast-cancer-predictor'

    endpoint_exists = False
    endpoints = sm.list_endpoints()
    if endpoints['Endpoints'] and len(endpoints['Endpoints']) > 0:
        endpoint_exists = any(True for endpoint in endpoints['Endpoints']
                              if endpoint['EndpointName'] == linear_endpoint)

    if endpoint_exists:
        print(f'Endpoint {linear_endpoint} exists, updating with new model')

        sm.update_endpoint(
            EndpointName=linear_endpoint,
            EndpointConfigName=linear_endpoint_config
        )
    else:
        print(f'Creating endpoint {linear_endpoint}')

        create_endpoint_response = sm.create_endpoint(
            EndpointName=linear_endpoint,
            EndpointConfigName=linear_endpoint_config)
        print(create_endpoint_response['EndpointArn'])

    resp = sm.describe_endpoint(EndpointName=linear_endpoint)
    status = resp['EndpointStatus']
    print("Endpoint Status: " + status)

    sm.get_waiter('endpoint_in_service').wait(EndpointName=linear_endpoint)

    resp = sm.describe_endpoint(EndpointName=linear_endpoint)
    status = resp['EndpointStatus']
    print("Arn: " + resp['EndpointArn'])
    print("Status: " + status)

    if status != 'InService':
        raise Exception('Endpoint creation did not succeed')

    return linear_endpoint


def test_endpoint(linear_endpoint, test_X, test_y, train_X, train_y):
    runtime = boto3.client('runtime.sagemaker')

    payload = np2csv(test_X)
    response = runtime.invoke_endpoint(EndpointName=linear_endpoint,
                                       ContentType='text/csv',
                                       Body=payload)

    '''
    Let's compare linear learner based mean absolute prediction errors from a baseline prediction which uses 
    majority class to predict every instance.
    '''
    result = json.loads(response['Body'].read().decode())
    test_pred = np.array([r['score'] for r in result['predictions']])

    test_mae_linear = np.mean(np.abs(test_y - test_pred))
    test_mae_baseline = np.mean(np.abs(test_y - np.median(train_y)))  ## training median as baseline predictor

    print("Test MAE Baseline :", round(test_mae_baseline, 3))
    print("Test MAE Linear:", round(test_mae_linear, 3))

    '''
    Let's compare predictive accuracy using a classification threshold of 0.5 for the predicted and compare against 
    the majority class prediction from training data set
    '''
    test_pred_class = (test_pred > 0.5) + 0;
    test_pred_baseline = np.repeat(np.median(train_y), len(test_y))

    prediction_accuracy = np.mean((test_y == test_pred_class)) * 100
    baseline_accuracy = np.mean((test_y == test_pred_baseline)) * 100

    print("Prediction Accuracy:", round(prediction_accuracy, 1), "%")
    print("Baseline Accuracy:", round(baseline_accuracy, 1), "%")


def np2csv(arr):
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()


parser = argparse.ArgumentParser()
# subparsers = parser.add_subparsers()

# init_parser = subparsers.add_parser('init')
parser.add_argument('--s3-backup-bucket', required=True)
parser.add_argument('--s3-sagemaker-bucket', required=True)
parser.add_argument('--role-arn', required=True)
parser.set_defaults(func=train_model_deploy)


if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)