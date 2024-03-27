# Hadoop Infrastructure with HDFS Role Only

This is a basic Hadoop infrastructure setup with HDFS role only, excluding mapreduce and yarn.

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

This setup allows you to utilize Hadoop's HDFS feature without mapreduce and yarn.
