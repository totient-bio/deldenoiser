FROM python:3.6.5

RUN mkdir /deldenoiser
RUN mkdir /deldenoiser/deldenoiser
RUN mkdir /deldenoiser/command-line-tool
COPY requirements.txt /deldenoiser

RUN pip install --upgrade pip
RUN pip install -r /deldenoiser/requirements.txt

COPY deldenoiser/__init__.py /deldenoiser/deldenoiser/
COPY deldenoiser/pbs_algorithm.py /deldenoiser/deldenoiser/
COPY deldenoiser/nullblockmodel.py /deldenoiser/deldenoiser/
COPY setup.py /deldenoiser
COPY README.md /deldenoiser
COPY command-line-tool/deldenoiser /deldenoiser/command-line-tool

RUN pip install /deldenoiser
