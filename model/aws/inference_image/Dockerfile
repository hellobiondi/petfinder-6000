# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

#Download an open source TensorFlow Docker image
FROM python:3.9-slim

# RUN wget -c https://ftp.gnu.org/gnu/glibc/glibc-2.29.tar.gz
# RUN tar -zxvf glibc-2.29.tar.gz
# RUN mkdir glibc-2.29/build
# WORKDIR glibc-2.29/build
# RUN ../configure --prefix=/opt/glibc
# RUN make && make install

RUN apt-get update \
&& apt-get install gcc nginx -y ca-certificates wget \
&& apt-get clean

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-training sagemaker-inference cornac pandas flask gunicorn numpy

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY server /opt/program
WORKDIR /opt/program
