
-- remove HTML from the raw_articles body

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
where body is not null and body != ''
