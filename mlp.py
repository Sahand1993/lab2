import tensorflow as tf
from Untitled import gen_datasets, add_noise

dataset = gen_dataset()

datasets["ysin_train"] = add_noise(datasets["ysin_train"])
datasets["ysin_test"] = add_noise(datasets["ysin_test"])

patterns_train = datasets["x_train"]
targets_train = datasets["ysin_train"]
patterns_test = datasets["x_test"]
targets_test = datasets["ysin_test"]

pattern = tf.placeholder(tf.float64)

node1 = tf.Variable()
node2 = tf.Variable()

node3 = tf.Variable()
node4 = tf.Variable()


