#!/bin/bash

# start spark cluster
/opt/spark/sbin/start-master.sh
/opt/spark/sbin/start-slave.sh spark://localhost:8080

sleep infinity