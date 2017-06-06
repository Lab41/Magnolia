FROM continuumio/anaconda3:4.2.0

RUN conda install -y flask
RUN pip install --upgrade pip
RUN apt-get install -y libsndfile1 && \
    pip install h5py

EXPOSE 5000
RUN pip install --upgrade matplotlib && pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp35-cp35m-linux_x86_64.whl
RUN pip install --upgrade scipy

ADD . /magnoliaWork
WORKDIR /magnoliaWork
RUN pip install .
WORKDIR /magnoliaWork/src/demo
