import boto3
import io
import csv
import datetime
import time
import os
from botocore.exceptions import ClientError

print('Loading function')

# Connect to Dynamo for scanning our table and S3 for sending a CSV version of our data
dynamo = boto3.client('dynamodb')
s3_client = boto3.client('s3')

table_name = os.environ['DDBTable']  # 'BreastCancerData'
fields = ["ID", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
          "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
          "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
          "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
          "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
          "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
throughput = int(os.environ['DDBThroughput'])
bucket = os.environ['BackupBucketName']


def extract_field(item, field):
    if field in item:
        return list(item[field].values())[0]
    else:
        return ''


def lambda_handler(event, context):

    try:

        print(f'Dumping DyanmoDB table {table_name} to S3 Bucket {bucket}')

        last_evaluated_key = None

        # Using eventual consistency so can double throughput
        # See https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.ProvisionedThroughput.html
        throughput_utilization = int(throughput * 0.8) * 2  # Only using 80% because we know no one is using the table

        # We will write the rows we process to an in-memory buffer, then format it as CSV and upload it to S3
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(fields)

        while True:

            # Keep track of time to try to not exceed our Provisioned Throughput
            start = time.time()

            try:
                if last_evaluated_key:
                    print(f'Getting batch starting from {last_evaluated_key}')
                    scanned_table = dynamo.scan(TableName=table_name,
                                                ExclusiveStartKey=last_evaluated_key,
                                                Limit=throughput_utilization,
                                                ConsistentRead=False)
                else:
                    print('Getting first batch')
                    scanned_table = dynamo.scan(TableName=table_name,
                                                Limit=throughput_utilization,
                                                ConsistentRead=False)

                items = scanned_table['Items']
                # Package the results into the column layout we will want for Sagemaker
                results = [[extract_field(item, field) for field in fields] for item in items]

                for res in results:
                    writer.writerow(res)

            except dynamo.exceptions.ProvisionedThroughputExceededException:
                print("EXCEEDED THROUGHPUT ON TABLE " + table_name)

            # Sleep if we need to to not exceed our throughput limitations
            end = time.time()
            diff = end - start
            if diff < 1:
                print('Sleeping to not exceed throughput')
                time.sleep(diff)

            # If LastEvaluatedKey doesn't exist, we're done
            try:
                last_evaluated_key = scanned_table["LastEvaluatedKey"]
            except KeyError:
                break

        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H:%M:%S_UTC')
        key = f'DynamoDB_Backup_{timestamp}'

        s3_client.put_object(Body=output.getvalue(), Bucket=bucket, Key=key)

        return 'Success'

    except ClientError as e:
        print(e)
    except ValueError as ve:
        print('ValueError:', ve)
    except Exception as ex:
        print(ex)


