# VEDA Black Marble
Veda Implementation of Black Marble Data Processing

## Notebook

You can use the '[bm-data-generator.ipynb](notebooks%2Fbm-data-generator.ipynb)' to generate black marble data products
for a given area of interest. If will download all the necessary input files from Landsat dataset, Openstreet Map and VNP46A2 night 
time data products archive before generating the final product. It is recommended to run this notebook in [VEDA Hub](https://hub.openveda.cloud/) 
as this runs inheriting the AWS credentials injected into VEDA Hub's notebook environment and you are not charged for data download
overhead from Landsat data which is significant.

## Docker image build and push commands

Following instructions are for setting up the Black Marble algorithm in Amazon ECS container environment. This is recommended 
for automated and repeated data generation workflows

```
cd docker
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 018923174646.dkr.ecr.us-west-2.amazonaws.com
docker build --platform linux/amd64 -t black-marble .
docker tag black-marble:latest 018923174646.dkr.ecr.us-west-2.amazonaws.com/black-marble:latest
docker push 018923174646.dkr.ecr.us-west-2.amazonaws.com/black-marble:latest
```