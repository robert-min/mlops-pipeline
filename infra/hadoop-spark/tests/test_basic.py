import pytest
from pyspark.sql import SparkSession
from pyspark.testing.utils import assertDataFrameEqual


@pytest.fixture
def spark():
    spark = SparkSession.builder.master("spark://localhost:8080").appName("SparkSQL").getOrCreate()
    yield spark


def test_single_space(spark):
    sample_data = [{"name": "John    D.", "age": 30},
                   {"name": "Alice   G.", "age": 25},
                   {"name": "Bob  T.", "age": 35},
                   {"name": "Eve   A.", "age": 28}]

    # Create a Spark DataFrame
    original_df = spark.createDataFrame(sample_data)

    # Apply the transformation function from before
    transformed_df = spark.remove_extra_spaces(original_df, "name")

    expected_data = [{"name": "John D.", "age": 30},
    {"name": "Alice G.", "age": 25},
    {"name": "Bob T.", "age": 35},
    {"name": "Eve A.", "age": 28}]

    expected_df = spark.createDataFrame(expected_data)

    assertDataFrameEqual(transformed_df, expected_df)
