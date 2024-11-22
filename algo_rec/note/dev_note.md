# feature_column
bucket-fc:
cate
-


# input-parse-s3file

# estimator


CREATE TABLE `algo`.`cn_rec_detail_sample_v0`(
    `sample_id` string COMMENT 'sample_id',
    `country` string COMMENT 'country',
    `uuid` string COMMENT 'uuid',
    `pssid` string COMMENT 'pssid',
    `goods_id` string COMMENT 'goods_id',
    `main_goods_id` string COMMENT 'main_goods_id',
    `ts` string COMMENT 'ts',
    `is_clk` string COMMENT 'is_clk',
    `is_pay` string COMMENT 'is_pay',
    `is_cart` string COMMENT 'is_cart',
    `is_wish` string COMMENT 'is_wish',
    `sales_price` int COMMENT 'sales_price',
    `goods_name` array < string > COMMENT 'goods_name',
    `main_color` string COMMENT 'main_color',
    `pic_id` bigint COMMENT 'pic_id',
    `cate_id` bigint COMMENT 'cate_id',
    `cate_name` string COMMENT 'cate_name',
    `category_id` bigint COMMENT 'category_id',
    `category` string COMMENT 'category',
    `cate_level1_id` bigint COMMENT 'cate_level1_id',
    `cate_level1_name` string COMMENT 'cate_level1_name',
    `cate_level2_id` bigint COMMENT 'cate_level2_id',
    `cate_level2_name` string COMMENT 'cate_level2_name',
    `cate_level3_id` bigint COMMENT 'cate_level3_id',
    `cate_level3_name` string COMMENT 'cate_level3_name',
    `cate_level4_id` bigint COMMENT 'cate_level4_id',
    `cate_level4_name` string COMMENT 'cate_level4_name',
    `splr_code` string COMMENT 'splr_code',
    `splr_name` string COMMENT 'splr_name',
    `byr_id` bigint COMMENT 'byr_id',
    `byr_name` string COMMENT 'byr_name',
    `site_inline_status_map` map < string,
    bigint > COMMENT 'site_inline_status_map',
    `prop_seaon` string COMMENT 'prop_seaon',
    `prop_length` string COMMENT 'prop_length',
    `prop_main_material` string COMMENT 'prop_main_material',
    `prop_pattern` string COMMENT 'prop_pattern',
    `prop_style` string COMMENT 'prop_style',
    `prop_quantity` string COMMENT 'prop_quantity',
    `prop_fitness` string COMMENT 'prop_fitness',
    `model_ids` string COMMENT 'model_ids',
    `main_model_id` string COMMENT 'main_model_id',
    `colors` array < string > COMMENT 'colors',
    `color_ids` array < int > COMMENT 'color_ids',
    `show_7d` int COMMENT 'show_7d',
    `click_7d` int COMMENT 'click_7d',
    `cart_7d` int COMMENT 'cart_7d',
    `ord_total` int COMMENT 'ord_total',
    `pay_total` int COMMENT 'pay_total',
    `ord_7d` int COMMENT 'ord_7d',
    `pay_7d` int COMMENT 'pay_7d',
    `ctr_7d` double COMMENT 'ctr_7d',
    `cvr_7d` double COMMENT 'cvr_7d'
) PARTITIONED BY (`ds` string COMMENT 'ds') ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' STORED AS INPUTFORMAT 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat' TBLPROPERTIES (
    'creation_platform' = 'coral',
    'is_core' = 'false',
    'is_starred' = 'false',
    'parquet.compression' = 'snappy',
    'primary_key' = '',
    'status' = '3',
    'transient_lastDdlTime' = '1730919652',
    'ttl' = '30',
    'ttlFirstExecTime' = '2024-11-07 03:00:52',
    'ttl_extension' = '{\"type\":\"mtime\"}'
)