# from feature_serv_describe_tfv2 import feature_describe_pred
import tensorflow as tf
import tensorflow.compat.v1 as v1

feature_describe_pred = {
    "pv_1d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_1d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_1d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_1d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_1d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_1d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_1d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_1d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_1d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pv_3d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_3d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_3d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_3d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_3d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_3d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_3d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_3d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_3d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pv_5d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_5d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_5d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_5d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_5d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_5d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_5d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_5d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_5d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pv_7d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_7d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_7d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_7d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_7d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_7d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_7d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_7d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_7d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pv_14d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_14d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_14d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_14d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_14d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_14d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pv_30d": v1.FixedLenFeature(1, tf.int64, -1),
    "ipv_30d": v1.FixedLenFeature(1, tf.int64, -1),
    "cart_30d": v1.FixedLenFeature(1, tf.int64, -1),
    "wish_30d": v1.FixedLenFeature(1, tf.int64, -1),
    "pay_30d": v1.FixedLenFeature(1, tf.int64, -1),
    "pctr_30d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcart_30d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pwish_30d": v1.FixedLenFeature(1, tf.float32, -1.0),
    "pcvr_30d": v1.FixedLenFeature(1, tf.float32, -1.0),
    # "mt_i2i_main_score": v1.FixedLenFeature(1, tf.float32, -1.0),
    # "mt_i2i_long_score": v1.FixedLenFeature(1, tf.float32, -1.0),
    # "mt_i2i_short_score": v1.FixedLenFeature(1, tf.float32, -1.0)
    #
    # , "mt_i2i_main": v1.FixedLenFeature(1, tf.int64, 0)
    # , "mt_i2i_long": v1.FixedLenFeature(1, tf.int64, 0)
    # , "mt_i2i_short": v1.FixedLenFeature(1, tf.int64, 0)
    # , "mt_hot_i2leaf": v1.FixedLenFeature(1, tf.int64, 0)
    # , "mt_hot": v1.FixedLenFeature(1, tf.int64, 0)
}
js_int = {}
js_float = {}
for name, v in feature_describe_pred.items():
    if name in ['is_clk', 'is_pay', 'sample_id']:
        continue
    if 'tf.int64' in str(v):
        js_int[name] = -1
    if 'tf.float32' in str(v):
        js_float[name] = -1

print(js_int)
print(js_float)
