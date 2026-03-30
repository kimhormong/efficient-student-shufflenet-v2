import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def get_flops_tf_keras(model, input_shape=(1, 224, 224, 3)):
    @tf.function
    def model_fn(x):
        return model(x, training=False)

    concrete = model_fn.get_concrete_function(
        tf.TensorSpec(input_shape, tf.float32)
    )
    frozen = convert_variables_to_constants_v2(concrete)
    graph_def = frozen.graph.as_graph_def(add_shapes=True)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=run_meta, cmd="op", options=opts
        )

    return flops.total_float_ops