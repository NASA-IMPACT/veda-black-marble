FROM python:3.11

RUN apt-get update

RUN apt-get install -y libgdal-dev

RUN pip3 install utm
RUN pip3 install scikit-image
RUN pip3 install 'gdal==3.6.2'
RUN pip3 install --no-cache-dir --force-reinstall 'GDAL[numpy]==3.6.2'
RUN pip3 install h5py
RUN pip3 install matplotlib
RUN pip3 install pyproj
RUN pip3 install ogr
RUN pip3 install shapely
RUN pip3 install boto3
RUN pip3 install rasterio
RUN pip3 install fiona
RUN pip3 install osmnx
RUN pip3 install geopy

RUN mkdir -p /root/outputs/final
RUN mkdir -p /root/outputs/temp

ADD WRS2_descending_0 /root/WRS2_descending_0/
ADD inferno_b.npy /root/inferno_b.npy
ADD inferno_g.npy /root/inferno_g.npy
ADD inferno_r.npy /root/inferno_r.npy
 
ADD bm.py /root/bm.py
WORKDIR /root
CMD ["python3", "/root/bm.py"]

