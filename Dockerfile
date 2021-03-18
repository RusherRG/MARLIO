FROM tensorflow/tensorflow

RUN apt-get update && \
    apt-get install -y \
    wget \
    curl \
    git \
    libasound2

RUN git clone https://9620248ce86c025673888441a744b21cbc0cfe4d@github.com/RusherRG/MARLIO
RUN git clone https://9620248ce86c025673888441a744b21cbc0cfe4d@github.com/Korusuke/MARLIO-runner

WORKDIR /MARLIO
RUN git checkout restructure && \
    pip3 install -r requirements.txt && \
    pip3 install -e .

WORKDIR /MARLIO-runner
RUN pip3 install -r requirements.txt

RUN mkdir ~/.MARLIO-runner