from collections import defaultdict
import os
import polars as pl

FOLD_COUNT = 10
CONST_N = 5

dataset_names = [
    'abt-buy', 'amazon-google', 'products_(Walmart-Amazon)',
    'wdc_xlarge_computers', 'wdc_xlarge_shoes', 'wdc_xlarge_watches'
]

def create_folds(source, N):
    positives = source.filter(
        pl.col('label')
    ).sample(fraction=1, shuffle=True, seed=0).with_row_count('fold').with_columns(
        pl.col('fold') % FOLD_COUNT
    )
    negatives = source.filter(
        ~pl.col('label')
    ).sample(fraction=1, shuffle=True, seed=0).head(
        positives.shape[0] * N
    ).with_row_count('fold').with_columns(
        pl.col('fold') % FOLD_COUNT
    )
    return pl.concat([positives, negatives])

data = defaultdict(defaultdict)
for key in dataset_names:
    path = f'data/{key}/feature_vector.csv'
    source = pl.scan_csv(
        path
    ).with_columns(
        pl.col('label').cast(pl.Boolean)
    ).collect()
    positives = source.lazy().filter(pl.col('label')).select(pl.count()).collect().item()
    negatives = source.shape[0] - positives
    N = min(negatives // positives, CONST_N)
    data[key]['data'] = create_folds(source, N)
    data[key]['N'] = N
    
amazon = pl.read_csv(
    'data/amazon-google/record_descriptions/2_google.csv',
    encoding='iso-8859-1'
)
google = pl.read_csv(
    'data/amazon-google/record_descriptions/1_amazon.csv',
    encoding='iso-8859-1'
)

def join_amazon_google(data):
    return data.join(
        amazon.fill_null(''),
        how='left',
        left_on='target_id',
        right_on='subject_id',
        validate='m:1'
    ).join(
        google.fill_null(''),
        how='left',
        left_on='source_id',
        right_on='subject_id',
        validate='m:1',
        suffix='_r'
    ).with_row_count(
        'idx'
    ).select(
        pl.col('idx'),
        (pl.col('name') + ' ' + pl.col('description') + ' ' + pl.col('manufacturer')).str.to_lowercase().alias('text_left'),
        (pl.col('name_r') + ' ' + pl.col('description_r') + ' ' + pl.col('manufacturer_r')).str.to_lowercase().alias('text_right'),
        pl.col('label').cast(pl.UInt8)
    )

abt = pl.read_csv(
    'data/abt-buy/record_descriptions/1_abt.csv',
    encoding='iso-8859-1',
    dtypes={'price': pl.Utf8}
)
buy = pl.read_csv(
    'data/abt-buy/record_descriptions/2_buy.csv',
    encoding='iso-8859-1',
    dtypes={'price': pl.Utf8}
)

def join_abt_buy(data):
    return data.join(
        abt.fill_null(''),
        how='left',
        left_on='source_id',
        right_on='subject_id',
        validate='m:1'
    ).join(
        buy.fill_null(''),
        how='left',
        left_on='target_id',
        right_on='subject_id',
        validate='m:1',
        suffix='_r'
    ).with_row_count(
        'idx'
    ).select(
        pl.col('idx'),
        (pl.col('name') + ' ' + pl.col('description') + ' ' + pl.col('price')).str.to_lowercase().alias('text_left'),
        (pl.col('name_r') + ' ' + pl.col('description_r') + ' ' + pl.col('price_r')).str.to_lowercase().alias('text_right'),
        pl.col('label').cast(pl.UInt8)
    )

amazon2 = pl.read_csv(
    'data/products_(Walmart-Amazon)/record_descriptions/2_amazon.csv',
    infer_schema_length=0,
    dtypes={'subject_id': pl.Int64}
).fill_null('').with_columns(
    pl.col(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
).select(
    pl.col('subject_id'),
    pl.concat_str(
        'title',
        'brand',
        'modelno',
        'price',
        'groupname',
        'longdescr',
        separator=' '
    ).alias('text')
)
walmart = pl.read_csv(
    'data/products_(Walmart-Amazon)/record_descriptions/1_walmart.csv',
    infer_schema_length=0,
    dtypes={'subject_id': pl.Int64}
).fill_null('').with_columns(
    pl.col(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
).select(
    pl.col('subject_id'),
    pl.concat_str(
        'title',
        'brand',
        'modelno',
        'price',
        'groupname',
        'longdescr',
        separator=' '
    ).alias('text')
)

def join_walmart_amazon(data):
    return data.join(
        walmart,
        how='left',
        left_on='source_id',
        right_on='subject_id',
        validate='m:1'
    ).join(
        amazon2,
        how='left',
        left_on='target_id',
        right_on='subject_id',
        validate='m:1',
        suffix='_r'
    ).with_row_count(
        'idx'
    ).select(
        pl.col('idx'),
        pl.col('text').alias('text_left'),
        pl.col('text_r').alias('text_right'),
        pl.col('label').cast(pl.UInt8)
    )

computers = pl.read_csv(
    'data/wdc_xlarge_computers/record_descriptions/1_computers_single_view.csv'
).fill_null('').with_columns(
    pl.col(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
).select(
    pl.col('subject_id'),
    pl.concat_str('title', 'brand', 'description', 'specTableContent', separator=' ').alias('text')
)

def join_computers(data):
    return data.join(
        computers,
        how='left',
        left_on='target_id',
        right_on='subject_id',
        validate='m:1'
    ).join(
        computers,
        how='left',
        left_on='source_id',
        right_on='subject_id',
        validate='m:1',
        suffix='_r'
    ).with_row_count(
        'idx'
    ).select(
        pl.col('idx'),
        pl.col('text').alias('text_left'),
        pl.col('text_r').alias('text_right'),
        pl.col('label').cast(pl.UInt8)
    )

shoes = pl.read_csv(
    'data/wdc_xlarge_shoes/record_descriptions/1_shoes_single_view.csv',
    encoding='iso-8859-1'
).fill_null('').with_columns(
    pl.col(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
).select(
    pl.col('subject_id'),
    pl.concat_str('title', 'brand', 'description', 'specTableContent', separator=' ').alias('text')
)

def join_shoes(data):
    return data.join(
        shoes,
        how='left',
        left_on='target_id',
        right_on='subject_id',
        validate='m:1'
    ).join(
        shoes,
        how='left',
        left_on='source_id',
        right_on='subject_id',
        validate='m:1',
        suffix='_r'
    ).with_row_count(
        'idx'
    ).select(
        pl.col('idx'),
        pl.col('text').alias('text_left'),
        pl.col('text_r').alias('text_right'),
        pl.col('label').cast(pl.UInt8)
    )

watches = pl.read_csv(
    'data/wdc_xlarge_watches/record_descriptions/1_watches_single_view.csv'
).fill_null('').with_columns(
    pl.col(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
).select(
    pl.col('subject_id'),
    pl.concat_str('title', 'brand', 'description', 'specTableContent', separator=' ').alias('text')
)

def join_watches(data):
    return data.join(
        watches,
        how='left',
        left_on='target_id',
        right_on='subject_id',
        validate='m:1'
    ).join(
        watches,
        how='left',
        left_on='source_id',
        right_on='subject_id',
        validate='m:1',
        suffix='_r'
    ).with_row_count(
        'idx'
    ).select(
        pl.col('idx'),
        pl.col('text').alias('text_left'),
        pl.col('text_r').alias('text_right'),
        pl.col('label').cast(pl.UInt8)
    )

def prepare_experiment(key, join_func):
    d = data[key]
    if key == 'products_(Walmart-Amazon)':
        key = 'walmart-amazon'

    for N in range(1, d['N'] + 1):
        for i in range(FOLD_COUNT):
            os.makedirs(f'entity-matching-transformer/data/{key}-{N}-{i}', exist_ok=True)
            test_fold_index = i
            validation_fold_index = (i - 1) % FOLD_COUNT

            d['data'].head(
                d['data'].shape[0] // (d['N'] + 1) * (N + 1) # sample a number of rows to get the right ratio
            ).filter(
                ~pl.col('fold').is_in([validation_fold_index, test_fold_index]) # remove validation and test fold data
            ).pipe(
                join_func
            ).write_csv(
                f'entity-matching-transformer/data/{key}-{N}-{i}/train.tsv',
                separator='\t'
            )

            d['data'].filter(
                pl.col('fold') == validation_fold_index
            ).pipe(
                join_func
            ).write_csv(
                f'entity-matching-transformer/data/{key}-{N}-{i}/dev.tsv',
                separator='\t'
            )

            d['data'].filter(
                pl.col('fold') == test_fold_index
            ).pipe(
                join_func
            ).write_csv(
                f'entity-matching-transformer/data/{key}-{N}-{i}/test.tsv',
                separator='\t'
            )

prepare_experiment('amazon-google', join_amazon_google)
prepare_experiment('abt-buy', join_abt_buy)
prepare_experiment('products_(Walmart-Amazon)', join_walmart_amazon)
prepare_experiment('wdc_xlarge_computers', join_computers)
prepare_experiment('wdc_xlarge_shoes', join_shoes)
prepare_experiment('wdc_xlarge_watches', join_watches)