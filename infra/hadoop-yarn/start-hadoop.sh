#!/bin/bash

# set enviroment
export HADOOP_HOME=/opt/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# start hadoop node
function start_namenode {
    echo "Starting Namenode..."
    hdfs namenode -format -force
    hdfs --config $HADOOP_CONF_DIR namenode
}

function start_datanode {
    echo "Checking Namenode status..."
    while true; do
        if hdfs dfsadmin -report &> /dev/null; then
            echo "Namenode is ready."
            break
        else
            echo "Waiting for Namenode to be ready..."
            sleep 3
        fi
    done

    echo "Starting Datanode..."
    hdfs --config $HADOOP_CONF_DIR datanode
}

function start_resourcemanager {
    echo "Starting ResourceManager..."
    yarn --config $HADOOP_CONF_DIR resourcemanager
}

function start_nodemanager {
    echo "Starting NodeManager..."
    yarn --config $HADOOP_CONF_DIR nodemanager
}


case $1 in
  namenode)
    start_namenode
    ;;
  datanode)
    start_datanode
    ;;
  resourcemanager)
    start_resourcemanager
    ;;
  nodemanager)
    start_nodemanager
    ;;
  *)
    echo "Unknown component: $1"
    exit 1
    ;;
esac

# Waiting for container not to terminate
tail -f $HADOOP_HOME/logs/*
