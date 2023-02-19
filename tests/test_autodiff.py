import tensorflow as tf
from tfga import GeometricAlgebra
from tfga.blades import BladeKind

algebra = GeometricAlgebra([1, 1, 1])


def get_rotor_loss(values):
    rotor = algebra.from_tensor_with_kind(values, BladeKind.BIVECTOR)
    s = algebra.geom_prod(rotor, algebra.reversion(rotor))[..., 0]
    return tf.reduce_sum(tf.math.square(s - 1))


def test_make_rotor():
    rotor_values = tf.Variable([1, 2, 3], dtype=tf.float32)

    optimizer = tf.optimizers.Adam(1)

    def train_step():
        with tf.GradientTape() as tape:
            tape.watch(rotor_values)
            loss = get_rotor_loss(rotor_values)

        grads = tape.gradient(loss, rotor_values)
        optimizer.apply_gradients(zip([grads], [rotor_values]))

    for _ in range(100):
        train_step()

    final_loss = get_rotor_loss(rotor_values)
    assert final_loss < 0.1
