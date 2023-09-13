#Download an open source TensorFlow Docker image
FROM python:3.9-slim

# RUN wget -c https://ftp.gnu.org/gnu/glibc/glibc-2.29.tar.gz
# RUN tar -zxvf glibc-2.29.tar.gz
# RUN mkdir glibc-2.29/build
# WORKDIR glibc-2.29/build
# RUN ../configure --prefix=/opt/glibc
# RUN make && make install

RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean

# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-training cornac

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
#COPY train.py /opt/ml/code/train.py

# Defines cifar10.py as script entrypoint
#ENV SAGEMAKER_PROGRAM train.py
