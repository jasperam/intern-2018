def step_fn(x):
    return 1 if x > 0 else 0

def step_gradient(x):
    return 0

import numpy as np
from tensorflow.python.framework import ops
import tensorflow as tf
step_np = np.vectorize(step_fn)
step_gradient_np = np.vectorize(step_gradient)
step_np_32 = lambda x: step_np(x).astype(np.float32)
step_gradient_np_32 = lambda x: step_gradient_np(x).astype(np.float32)

def step_gradient_tf(x, name=None):
    with ops.name_scope(name, "step_gradient_tf", [x]):
        y = tf.py_func(step_gradient_np_32, [x], [tf.float32], name=name, stateful=False)
        return y[0]

def my_py_func(func, inp, Tout, stateful=False, name=None, my_gradient_func=None):
    # need to generate a unique name to avoid duplicates:
    random_name = "PyFuncGrad" + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(random_name)(my_gradient_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": random_name, "PyFuncStateless": random_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def _step_gradient(op, pred_grad):
    x = op.inputs[0]
    cur_grad = step_gradient(x)
    next_grad = pred_grad * cur_grad
    return next_grad

def step_tf(x, name=None):
    with ops.name_scope(name, "step_tf", [x]) as name:
        y = my_py_func(step_np_32,
                       [x],
                       [tf.float32],
                       stateful=False,
                       name=name,
                       my_gradient_func=_step_gradient)
    return y[0]



