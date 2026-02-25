{{ config(materialized='table') }}

WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY search_term
            ORDER BY hash(CAST(id AS VARCHAR) || 'seed_2026')
        ) AS rn
    FROM {{ ref('cleaned_articles') }}
)

SELECT *
FROM ranked
WHERE rn <= 10