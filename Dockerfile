FROM python:3.10-buster

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -r requirements.txt
