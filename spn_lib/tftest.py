#test_file
import tensorflow as tf

a = tf.Variable([[1, 2, 3, 4], [2, 3, 4, 1]], dtype=tf.float64)
m = tf.Variable([1, 2, 3, 4], dtype=tf.float64)
n = tf.constant([0, 0, 1, 1])
s = tf.Session()
i = tf.initialize_all_variables
s.run(i())
t = lambda x: tf.transpose(x)
b = tf.mul(t(tf.segment_max(t(a), n)), 1.99)
g = t(tf.gather(t(b), n))
c = tf.round(tf.div(a, g))
d = tf.reduce_sum(c, reduction_indices=0)
mm = tf.mul(a, m)
print s.run(mm)