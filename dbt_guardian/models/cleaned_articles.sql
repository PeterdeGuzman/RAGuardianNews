select
    *,
    regexp_replace(body, '<[^>]+>', '', 'g') as clean_body
from {{ source('raw', 'raw_articles') }}