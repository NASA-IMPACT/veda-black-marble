# veda-black-marble
Veda Implementation of Black Marble Data Processing

## Docker image build and push commands
```
cd docker
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 018923174646.dkr.ecr.us-west-2.amazonaws.com
docker build --platform linux/amd64 -t black-marble .
docker tag black-marble:latest 018923174646.dkr.ecr.us-west-2.amazonaws.com/black-marble:latest
docker push 018923174646.dkr.ecr.us-west-2.amazonaws.com/black-marble:latest
```