import boto3
import pprint


class BlackMarbleRunner:

    def __init__(self, access_key, secret_key, earth_data_token,
                 task_arn = 'arn:aws:ecs:us-west-2:018923174646:task-definition/black-marble-executions:6',
                 cluster ='BMWorkloadCluster', subnets = [
                        'subnet-6c1d0c15',
                        'subnet-6304e43e',
                        'subnet-28724063',
                        'subnet-32d44f19'
                    ], security_groups = [
                        'sg-a50d4de4',
                    ]):
        self.access_key = access_key
        self.secret_key = secret_key
        self.earth_data_token = earth_data_token
        self.task_arn = task_arn
        self.cluster = cluster
        self.subnets = subnets
        self.security_groups = security_groups


    def run_bm_task(self, lat1, lat2, long1, long2, year, month, day,
                    cpu = 8, memory = 32, memory_reservation = 24):

        ecs_client = boto3.client('ecs', aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key, region_name='us-west-2')

        response = ecs_client.run_task(
            cluster=self.cluster,
            count=1,
            launchType='FARGATE',
            taskDefinition=self.task_arn,
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': self.subnets,
                    'securityGroups': self.security_groups,
                    'assignPublicIp': 'ENABLED'
                }
            },
            overrides={
                'containerOverrides': [
                    {
                        'name': 'black-marble',
                        'environment': [
                            {
                                'name': 'LAT1',
                                'value': lat1
                            },
                            {
                                'name': 'LAT2',
                                'value': lat2
                            },
                            {
                                'name': 'LONG1',
                                'value': long1
                            },
                            {
                                'name': 'LONG2',
                                'value': long2
                            },
                            {
                                'name': 'EARTH_DATA_TOKEN',
                                'value': self.earth_data_token
                            },
                            {
                                'name': 'AWS_ACCESS_KEY',
                                'value': self.access_key
                            },
                            {
                                'name': 'AWS_ACCESS_SECRET',
                                'value': self.secret_key
                            },
                            {
                                'name': 'YEAR',
                                'value': year
                            },
                            {
                                'name': 'MONTH',
                                'value': month
                            },
                            {
                                'name': 'DAY',
                                'value': day
                            },
                        ],
                        'cpu': cpu * 1024,
                        'memory': memory * 1024,
                        'memoryReservation': memory_reservation * 1024,
                    },
                ],
            }
        )
        return response['tasks'][0]['taskArn']

    def stop_bm_task(self, task_arn):

        ecs_client = boto3.client('ecs', aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key, region_name='us-west-2')

        response = ecs_client.stop_task(
            cluster=self.cluster,
            task=task_arn,
            reason='Manually stopping'
        )