import boto3
import pandas as pd
import decimal
import sys

# boto3 is the AWS SDK library for Python.
# The "resources" interface allow for a higher-level abstraction than the low-level client interface.
# More details here: http://boto3.readthedocs.io/en/latest/guide/resources.html
dynamo = boto3.resource('dynamodb')
table_name = 'BreastCancerData'
table = dynamo.Table(table_name)
data = pd.read_csv('breast_cancer_data_A.csv', delimiter=',')


# Dynamo expects Decimal type numbers, so we need to convert our floats
def convert_float_type_to_decimal(val):
    if type(val) is float:
        return decimal.Decimal(str(val))
    else:
        return val


print('Importing first half of data into Dynamo')

try:
    # Using batch_writer to automatically handle buffering and sending items in batches
    with table.batch_writer() as batch:
        for index, row in data.iterrows():
            payload = {key: convert_float_type_to_decimal(val)
                       for key, val in row.to_dict().items()}
            batch.put_item(Item=payload)
except dynamo.exceptions.ResourceNotFoundException:
    print(f'Table {table_name} was not found, make sure that the SagemakerStack.yml script ran successfully')
    sys.exit()
except dynamo.exceptions.ProvisionedThroughputExceededException:
    print('Provisioned throughput exceeded, try temporarily raising this value in the console')
    sys.exit()

print('Success')
