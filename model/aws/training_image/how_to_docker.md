# How to build Docker image

## Building Image
1. Navigate to /model/aws/training/base_image on your local machine where the Dockerfile is located.
2. Ensure that the docker daemon is running.
3. Run `docker build -t <image name> .`
4. Run `docker images` to view the built image.

## Push to AWS ECR
1. Install and configure AWS CLI on your local machine if you have not done so.
2. Authenticate with the AWS ECR private registry using `aws ecr get-login-password --region ap-southeast-1 | docker
   login --username AWS
   --password-stdin 418542404631.dkr.ecr.ap-southeast-1.amazonaws.com`
3. Run `docker images` and locate the ID of the image that you intend to push to ECR.
4. Run `docker tag <ID> 418542404631.dkr.ecr.ap-southeast-1.amazonaws.com/petfinder6000:<new tag>` to retag the
   image.
5. Run `docker push 418542404631.dkr.ecr.ap-southeast-1.amazonaws.com/petfinder6000:<new tag>` to push the image to ECR.
