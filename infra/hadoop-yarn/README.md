# Hadoop Infrastructure with ALL

This is a Hadoop infrastructure setup including mapreduce and Yarn.

## Execution

```bash
sudo docker build -t hadoop:0.0.0 .

sudo docker compose up -d
```

## Verification

1. Connect to the namenode using `docker exec`
2. Create a file to be stored in Hadoop
3. Store the file

```bash
# Check the path
hdfs dfs -ls

# Create a directory
hdfs dfs -mkdir /user

# Store the file
hdfs dfs -put /{file_name} /user/{file_name}
```

