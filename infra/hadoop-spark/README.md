# Spark Infrastructure with Hadoop

This is a Spark infrastructure with Hadoop.

## Execution
- Assume hadoop:0.0.0 image has already been created
- If not created, refer to infra/hadoop-yarn

```bash
sudo docker build -t spark:0.0.0 .

sudo docker compose up -d
```

## Verification

```bash
pyspark

spark-submit
```

