
-- remove HTML from the raw_articles body
--filter to only search_term is "artificial intelligence" or "generative AI"

select
    *,
    regexp_replace(body, '<[^>]+>', '', 'g') as clean_body
from {{ source('raw', 'raw_articles') }}
where search_term in ('artificial intelligence', 'generative AI')
