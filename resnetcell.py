import time

import tensorflow as tf

class DenseResNetCell(tf.keras.Model):
    def __init__(self, units):
        super().__init__()

        units1, units2 = units
        self.dense_a = tf.keras.layers.Dense(units1)
        self.dense_b = tf.keras.layers.Dense(units2)

    def __call__(self, input_tensor):
        x = self.dense_a(input_tensor)
        x = tf.nn.relu(x)

        x = self.dense_b(input_tensor)
        x = tf.nn.relu(x)

        return x + input_tensor

def build_dense_resnet_cell(m, n, k, output_dir):
    with tf.Graph().as_default() as g:
        with tf.device('device:gpu:0'):
            input_tensor = tf.Variable(tf.random.uniform([m, k], 0, 1))

            resnet_cell = DenseResNetCell([n, k])
            output_tensor = resnet_cell(input_tensor)

            with tf.control_dependencies([output_tensor]):
                pull = tf.no_op()

        init = tf.compat.v1.global_variables_initializer()

    tf.io.write_graph(
        g, output_dir, 'model.pb', as_text=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=1024)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--k', type=int, default=1024)
    parser.add_argument('-o', help='Output directory', dest='output_dir', required=True)

    ns = parser.parse_args()

    build_dense_resnet_cell(ns.m, ns.n, ns.k, ns.output_dir)
