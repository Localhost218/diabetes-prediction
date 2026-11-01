# Postgres 17 + Apache AGE (đồ thị tri thức) + pgvector (vector RAG)
FROM apache/age:release_PG17_1.6.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-17-pgvector \
    && rm -rf /var/lib/apt/lists/*

COPY init.sql /docker-entrypoint-initdb.d/01-extensions.sql
