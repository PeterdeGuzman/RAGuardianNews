{{ config(materialized='table') }}

WITH exploded AS (
    SELECT
        *,
        trim(unnest(string_split(search_terms, ','))) AS search_term
    FROM {{ ref('cleaned_articles') }}
),

ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY search_term
            ORDER BY hash(CAST(id AS VARCHAR) || 'seed_2026')
        ) AS rn
    FROM exploded
)

SELECT *
FROM ranked
WHERE rn <= 10