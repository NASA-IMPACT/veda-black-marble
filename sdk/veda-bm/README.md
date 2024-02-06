### Running Black Marble HD image generation on AWS Elastic Container Service


#### Launch image generation job
```
from veda_bm import BlackMarbleRunner

from veda_bm import BlackMarbleRunner
from veda_bm import OperationalCredentials
from veda_bm import DestinationStorageCredentials
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# AWS Access keys to launch containers
access_key = ''
secret_key = ''
earth_data_token = ''

lat1,long1,lat2,long2 = 33.034682, 21.279194, 32.465949, 22.919832
year,month,day = 2023, 1, 24

upload_bucket = 'veda-data-store-staging'
bucket_prefix = 'hd-blackmarble-nightlights-temp/' # This is a temp location for testing purposes

aws_region = os.environ['AWS_REGION']
aws_role_arn = os.environ['AWS_ROLE_ARN']
aws_sts_regional_endpoints = os.environ['AWS_STS_REGIONAL_ENDPOINTS']
with open(os.environ['AWS_WEB_IDENTITY_TOKEN_FILE']) as f:
   aws_id_token = f.read()

operational_credentials = OperationalCredentials(access_key = access_key,
                                             secret_key = secret_key,
                                             earth_data_token = earth_data_token)

upload_credentials = DestinationStorageCredentials(aws_region = aws_region,
                                                           aws_role_arn = aws_role_arn,
                                                           aws_id_token = aws_id_token,
                                                           aws_sts_regional_endpoints = aws_sts_regional_endpoints)

bm_runner = BlackMarbleRunner(operational_credentials, upload_credentials)

upload_file = bm_runner.get_upload_file_name(lat1 = lat1, lat2 = lat2,
                                          long1 = long1, long2 = long2,
                                          year = year, month = month, day = day,
                                          prefix = bucket_prefix)

task_arn = bm_runner.run_bm_task(lat1 = lat1, lat2 = lat2, long1 = long1, long2 = long2,
                             year = year, month = month, day = day,
                             upload_bucket = upload_bucket,
                             upload_file = upload_file)

status = bm_runner.get_bm_task_status(task_arn)
```

#### Kill an existing job

```
bm_runner.stop_bm_task(task_arn)
```