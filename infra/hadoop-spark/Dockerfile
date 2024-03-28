FROM hadoop:0.0.0

USER root

# install spark
ARG SPARK_VERSION=3.0.0
RUN wget -q https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop2.7.tgz && \
    tar -xzf spark-$SPARK_VERSION-bin-hadoop2.7.tgz -C /usr/local/ && \
    rm spark-$SPARK_VERSION-bin-hadoop2.7.tgz
RUN ln -s /usr/local/spark-$SPARK_VERSION-bin-hadoop2.7 /opt/spark

ENV SPARK_HOME /opt/spark
ENV PATH $SPARK_HOME/bin:$PATH

EXPOSE 8080 7077 4040

# add run script
COPY start-spark.sh /start-spark.sh
RUN chmod +x /start-spark.sh

CMD [ "sleep", "infinity" ]
