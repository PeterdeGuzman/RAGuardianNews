
-- remove HTML from the raw_articles body
--filter to only search_term is "artificial intelligence" or "generative AI"

select
    *,
    trim(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        regexp_replace(
                            body,
                            '<script[^>]*>.*?</script>', '', 'gis'
                        ),
                        '<style[^>]*>.*?</style>', '', 'gis'
                    ),
                    '<[^>]*>', '', 'g'
                ),
                '&[^;\s]+;', '', 'g'
            ),
            '\b(div|span|p|block|class|id|header|footer)\b', '', 'gi'
        )
    ) as clean_body
from {{ source('raw', 'raw_articles') }}
where search_term in ('artificial intelligence', 'generative AI')