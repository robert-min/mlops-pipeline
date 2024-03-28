FROM centos:7

USER root

# install libaray
RUN yum clean all
RUN rpm --rebuilddb
RUN yum install -y initscripts curl which tar sudo openssh-server openssh-clients nc perl snappy snappy-devel wget
RUN yum update -y libselinux

# install java
RUN yum -y update && yum -y install java-1.8.0-openjdk-devel
ENV JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk
ENV JRE_HOME=/usr/lib/jvm/java-1.8.0-openjdk/jre
ENV PATH=$PATH:$JAVA_HOME/bin:$JRE_HOME/bin

# hadoop install 
ARG HADOOP_VERSION=2.10.2	
# RUN curl -O --insecure https://dist.apache.org/repos/dist/release/hadoop/common/KEYS
# RUN gpg --import KEYS
# RUN gpg –verify hadoop-$HADOOP_VERSION-src.tar.gz.asc

RUN wget --no-check-certificate https://archive.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz \
    && tar xvfz hadoop-$HADOOP_VERSION.tar.gz \
    && mv hadoop-$HADOOP_VERSION /opt/hadoop-$HADOOP_VERSION \
    && rm hadoop-$HADOOP_VERSION.tar.gz

RUN cd /opt/hadoop-$HADOOP_VERSION && ln -s /opt/hadoop-$HADOOP_VERSION /opt/hadoop
RUN ln -s /opt/hadoop-$HADOOP_VERSION/etc/hadoop /etc/hadoop

ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=/etc/hadoop
ENV MULTIHOMED_NETWORK=1
ENV PATH $HADOOP_HOME/bin/:$PATH

RUN mkdir -p $HADOOP_HOME/logs && chmod -R 777 $HADOOP_HOME/logs
RUN mkdir -p /hadoop/dfs/name && mkdir -p /hadoop/dfs/data

# hadoop config setting
COPY ./config/core-site.xml ${HADOOP_CONF_DIR}/core-site.xml
COPY ./config/hdfs-site.xml ${HADOOP_CONF_DIR}/hdfs-site.xml
COPY ./config/mapred-site.xml ${HADOOP_CONF_DIR}/mapred-site.xml
COPY ./config/yarn-site.xml ${HADOOP_CONF_DIR}/yarn-site.xml

# add run script
COPY start-hadoop.sh /start-hadoop.sh
RUN chmod +x /start-hadoop.sh

# HDFS ports
EXPOSE 50070 50075 50010 50020 8020 9000 9870 8088

CMD ["/start-hadoop.sh"]
# CMD ["sleep", "infinity" ]
