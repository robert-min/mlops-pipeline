#!/bin/bash

# 환경 변수 설정
export HADOOP_VERSION=2.7.3
export HADOOP_HOME=/opt/hadoop-$HADOOP_VERSION
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# 인자에 따라 Namenode 또는 Datanode 시작
case $1 in
  namenode)
    echo "Starting Namenode..."
    hdfs namenode -format -force
    hdfs --config $HADOOP_CONF_DIR namenode
    ;;
  datanode)
    echo "Waiting for Namenode..."
    while ! nc -z namenode 9000; do sleep 1; done
    echo "Starting Datanode..."
    hdfs --config $HADOOP_CONF_DIR datanode
    ;;
  *)
    echo "Unknown argument: $1"
    exit 1
    ;;
esac

# 컨테이너가 종료되지 않도록 대기
tail -f $HADOOP_HOME/logs/*
