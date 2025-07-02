from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import IntegerType

def main():
    spark = SparkSession.builder \
        .config("spark.driver.memory", "2g") \
        .master("local[*]") \
        .getOrCreate()

    df = spark.read.csv("./DPRO-46/covid-data.csv", header=True, inferSchema=True)

    df_task_1 = df \
        .select(
            'iso_code'
            , 'location'
            , ((F.col('total_cases') - F.col('total_deaths'))/F.col('population') * 100).alias('% переболевших')) \
        .where((F.col('date') == '2020-03-31') & (F.length(F.col("iso_code")) == 3)) \
        .withColumn("% переболевших", F.round("% переболевших", 2)) \
        .orderBy(F.col('% переболевших') \
        .desc()).limit(15)

    print('15 стран с наибольшим процентом переболевших на 31 марта')
    df_task_1.show()

    w = Window.partitionBy("location").orderBy(F.col('new_cases').desc())

    df_task_2 = df \
        .select(
            'date'
            , 'location'
            , 'new_cases') \
        .where(F.to_date(F.col('date')).between('2021-03-25', '2021-03-31') & (F.length(F.col("iso_code")) == 3)) \
        .withColumn('rank', F.row_number().over(w)) \
        .filter(F.col("rank") == 1) \
        .drop("rank") \
        .sort(F.desc(F.col('new_cases'))) \
        .limit(10)

    print('Top 10 стран с максимальным зафиксированным кол-вом новых случаев за последнюю неделю марта 2021 в отсортированном порядке по убыванию')
    df_task_2.show()

    w = Window.partitionBy("location").orderBy(F.col('date').asc())

    df_task_3 = df \
    .filter(F.lower(F.col("location")).contains('russ')) \
    .where(F.to_date(F.col('date')).between('2021-03-25', '2021-03-31')) \
    .withColumn('prev_day', F.lag(F.col('new_cases'), 1, 0).over(w)) \
    .withColumn('diff', (F.col('new_cases') - F.col('prev_day')).cast(IntegerType())) \
    .select(
        F.col('date').alias('Дата')
        , F.col('prev_day').alias('Кол-во новых случаев вчера').cast(IntegerType())
        , F.col('new_cases').alias('Кол-во новых случаев сегодня').cast(IntegerType())
        , F.col('diff').alias('Дельта')
    )

    print('Изменение случаев относительно предыдущего дня в России за последнюю неделю марта 2021.')
    df_task_3.show()

if __name__ == "__main__":
    main()