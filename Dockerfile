FROM pytorch/pytorch

WORKDIR /home/apeksha/Projects/masters
ENV DEBIAN_FRONTEND=NONINTERACTIVE
RUN apt-get update -y && apt-get upgrade -y
RUN apt install git build-essential cmake python3-opencv -y
RUN pip install --upgrade pip
RUN pip3 install matplotlib scipy pandas Pillow \
    opencv-python opencv-contrib-python fastai timm jupyterlab notebook ipywidgets onnx torchinfo
