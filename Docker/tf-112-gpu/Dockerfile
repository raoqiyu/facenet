FROM tensorflow/tensorflow:1.12.0-gpu-py3

MAINTAINER Qiyu Rao <raoqiyu@enn.cn>

RUN rm -f /etc/apt/sources.list.d/*
COPY sources.list /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
		vim \
		less \
		iputils-ping \
		htop \
		byobu \
		python-qt4 \
		nodejs

# install jupyter lab
RUN pip3 install --upgrade pip
RUN pip3 install -U numpy pandas matplotlib scikit-learn
RUN pip3 install tqdm networkx opencv-pythuon h5py Pillow requests psutil scipy==1.2.1 jupyterlab==0.32.1



# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Set jupyter default shell
ENV SHELL=/bin/bash
ENV LANG=C.UTF-8

# set sklearn multi-processing ENV
RUN mkdir -p /tmp/joblib/
ENV JOBLIB_TEMP_FOLDER=/tmp/joblib/

WORKDIR "/tmp"

ADD . /var/tf-k8s
# Container entry point
ENTRYPOINT ["/var/tf-k8s/start.sh"]
