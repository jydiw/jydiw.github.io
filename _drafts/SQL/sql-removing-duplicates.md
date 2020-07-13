---
title: "Exploring and Preparing your Data with BigQuery"
excerpt_separator: "<!--more-->"
categories:
  - Google Cloud Platform
tags:
  - BigQuery
  -
---

key message

key pitfall

# Analytics Challenges

1. performance
   - queries taking too long
   - no easy way to combine and query all data
2. infrastructure
   - unsure how to maintain servers
   - on-premise clusters not scaling with performance
3. cost
   - can only afford to store a subset of data
   - do not have access to central data analytics tools

bigquery: web UI for sql queries on gcp

4.35 GB data takes about 6.7s to process with GCP

end state: processing massive amounts of data at speed

# reasons why GCP is used for data analysis

- storage is cheap (2c/GB/mo)
- focus on queries, not infrastructure
- massive scalability

**reduced cost compared to on-premise infrastructure**

traditional big data platforms require an investment in infrastructure:
1. storage
2. on-premise clusters
3. network access
4. maintenance staff

GCP takes infrastructure and security out of the question so that analysts can focus on querying

- on-demand scalability
  - under/over-provisioned
- **decoupling of storage and computing power**
- KEY: bigquery scales automatically and you only pay for what you use

## GCP basics:

### Projects
- organize and govern your activities in the cloud

### Resources
- storage
  - buckets are scalable containers for whatever data
- datasets

### Billing
- only billed for resources that you use

# data analyst tasks

1. ingest
2. transform
3. store
4. analyze
5. visualize

challenges:

challenges jpg

tools:

tools jpg

# bigquery tips and tricks

public datasets: https://cloud.google.com/bigquery/public-data/

metadeta about dataset
how to query
**what data is included**

you can use public datasets to get sample queries

before you run:

1. check your validator
   1. checks your code
   2. tells you how much data you're querying
   3. (you get 5TB free per month)
2. show options
   1. toggle off "Use Legacy SQL"
   2. #standardSQL enforces standard SQL
3. save query (give it a meaningful name)

caches on a per-user basis so previous queries are very fast

how do i quickly flip back to the data table?

click cmd (or windows? is that a mistake? ctrl?) it highlights datasets you're pulling queries from

https://bigquery.cloud.google.com/dataset/{project}:{dataset}

bigquery-public-data (longer list of public data)

Public Datasets (curated data?)

if you want to access your previous queries:

1. compose query
2. pull up saved query or
3. look through query history

"format query" will neatify your SQL code

There is a preview button without having to run a query

# fundamental bigquery takeaways
1. don't manage infrastructure
2. focus on finding insights from your data

bigquery is a petabyte scale data analytics warehouse
1.  fully-managed data warehouse
2.  reliable
3.  economical, pay-as-you-use
4.  secure
5.  auditable
6.  scalable
7.  flexible, able to pull from multiple datasets
8.  easy to use
9.  public

data architecture jpg

1. BigQuery Analytics Engine
2. BigQuery Managed Storage

data-roles jpg

1. data analyst: ingest, analyze, vizualize, querying
2. data scientist: analyze
3. data engineer: builds and maintains processing systems

gaming example

gaming jpg

# beyond the BigQuery Web UI

Google Cloud Platform supports Jupyter Notebooks as well.

To use BigQuery with Jupyter (or Python in general):

```python
from google.cloud import bigquery

client = bigquery.Client()
query_job = client.query("""
    {here is our SQL query}
""")

results = query_job.result()
```

# options for exploring datasets

## SQL + Web UI
1. flexible, fast, familiar
2. SQL is an imperative skill to have as an analyst

Steps to explore data through SQL:
1. ask good questions
2. know your data
3. write good SQL

# Query basics

- add #standardSQL as a comment in the first line
- show options -> disable legacy SQL
- keep this reference handy: https://cloud.google.com/bigquery/docs/reference#standard-sql-reference

if you're running multiple queries, highlight a portion of the query and "run selected".

we need to `escape` the table name using backticks.

# Function basics

```SQL
SELECT
    FORMAT("%'d", totrevenue) AS revenue
FROM
    `bigquery-public-data.irs_990.irs_990_2015`
ORDER BY
    totrevenue DESC
LIMIT
    10
```

best to save stylistic formatting for visualization so that data types don't change around too much

pitfall: aliases do not exist yet when filtering in WHERE since SQL is not done top-down

pitfall: being deliberate with which columns to return can greatly help with query speed

## aggregation functions

SUM
AVG
COUNT
MAX
MIN

ROUND

when you use aggregate functions, IMMEDIATELY include GROUP BY

```SQL
#standardSQL
SELECT
    ein                     # not aggregated
  , COUNT(ein) AS ein_count # aggregated
FROM
    `bigquery-public-data.irs_990.irs_990_2015`
GROUP BY
    ein
HAVING
    ein_count > 1
ORDER BY
    ein_count DESC;
```

WHERE filters rows pre-aggregation
HAVING filters rows post-aggregation, which allows for aliasing

## data types

data types jpg

CAST()
SAFE_CAST()

```SQL
SAFE_CAST("12345" AS INT64)
```

NULLs are valid values: absence of data or an empty set. NOT THE SAME as blank string value

NULLs not equal to anything, so we have to use IS operator

```SQL
#standardSQL
SELECT
    ein
  , street
  , city
  , state
  , zip
FROM
    `bigquery-public-data.irs_990.irs_990_ein`
WHERE
    state IS NULL
LIMIT
    10;
```

## date functions

date functions jpg

PARSE_DATETIME()

## string functions

## wildcard filters

% - any number of characters
_ - one character


Quiz 3:

Why shouldn't you use SELECT * to explore your dataset?

- BigQuery provides a "preview data" table feature that is faster (cached) and free to use
- Selecting all columns is computationally expensive--especially with no filters
- Selecting all columns, even with WHERE clause filters, will scan your entire dataset and incur charges for all bytes processed. This is a pitfall when returning potentially large columns (eg: long string fields)

Which one of these SQL clauses cannot operate against an alias that was just defined in your SELECT statement?

- WHERE

What is the core principle behind how BigQuery can effectively scale to process billions of rows of data?

- massive parallel processing of queries across distributed resources

