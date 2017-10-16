import tensorflow as tf
import numpy as numpy

# Tensorflow feature_column 有的numeric_column
f = [tf.feature_column.numeric_column('x',shape=[1])]

# Estimator 中的 LinearRegressor
estimator = tf.estimator.LinearRegressor(feature_columns=f)

# Tensorflow data sets
x_train = numpy.array([1,2,3,4])
y_train = numpy.array([0,-1,-2,-3])
# evaluation
x_eval = numpy.array([2,5,8,1])
y_eval = numpy.array([-1.01,-4.1,-7,0])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x':x_train},y_train,batch_size=4,num_epochs=None,shuffle=True
)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False
)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x':x_eval},y_eval,batch_size=4,num_epochs=1000,shuffle=False
)

# Begin training
estimator.train(input_fn=input_fn,steps=1000)

# Evaluation
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train_metrics: %r"%train_metrics)
print("eval_metrics: %r"%eval_metrics)