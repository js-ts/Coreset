FROM python:3.8

RUN apt-get -y update && apt-get -y install osmium-tool && apt update && apt-get -y install libpq-dev gdal-bin libgdal-dev libxml2-dev libxslt-dev

ADD Coreset Coreset

ADD monaco-latest.geojson .

RUN cd Coreset && pip3 install -r requirements.txt