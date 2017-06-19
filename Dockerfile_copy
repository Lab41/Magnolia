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

RUN mkdir /magnoliaWork/src/demo/app/static/models && \
    echo "Now getting Deep Clustering code" && \
    wget "https://www.dropbox.com/s/01f1jrss5leqarq/deep_clustering.ckpt.data-00000-of-00001?dl=0" -O \
         /magnoliaWork/src/demo/app/static/models/deep_clustering.ckpt.data-00000-of-00001 && \
    wget "https://www.dropbox.com/s/jeyjsgo4eg1e7uz/deep_clustering.ckpt.index?dl=0" -O \
         /magnoliaWork/src/demo/app/static/models/deep_clustering.ckpt.index && \
    wget "https://www.dropbox.com/s/w56skx9k5plz316/deep_clustering.ckpt.meta?dl=0" -O \
         /magnoliaWork/src/demo/app/static/models/deep_clustering.ckpt.meta && \
    echo "Now getting Lab41 Model" && \
    wget "https://www.dropbox.com/s/yrct5fmtxh9xisv/lab41_nonorm-final.ckpt.data-00000-of-00001?dl=0" -O \
         /magnoliaWork/src/demo/app/static/models/lab41_nonorm-final.ckpt.data-00000-of-00001 && \
    wget "https://www.dropbox.com/s/oa05j8mrfenmohf/lab41_nonorm-final.ckpt.index?dl=0" -O \
         /magnoliaWork/src/demo/app/static/models/lab41_nonorm-final.ckpt.index && \
    wget "https://www.dropbox.com/s/xka5emn0fbfzvq7/lab41_nonorm-final.ckpt.meta?dl=0" -O \
         /magnoliaWork/src/demo/app/static/models/lab41_nonorm-final.ckpt.meta

RUN mkdir /magnoliaWork/src/demo/app/resources

