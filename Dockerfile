ARG BASE=pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM ${BASE}

RUN apt-get update

RUN apt-get install nano

RUN apt-get install git

RUN pip install omegaconf==2.3.0 librosa==0.10.2.post1 tensorboard==2.17.0 matplotlib==3.9.1 jupyter==1.0.0 transformers==4.43.3 torchcrepe==0.0.23 pandas

WORKDIR /root

