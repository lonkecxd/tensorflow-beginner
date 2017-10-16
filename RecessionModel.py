import tensorflow as tf
sess = tf.Session()

# Parameters
W = tf.Variable([.3],dtype=tf.float32)
b = tf.Variable([-.3],dtype=tf.float32)

# Input x, and Expected Output y
x = tf.placeholder(tf.float32)
model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.squared_difference(model,y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_data = [1,2,3,4]
y_data = [0,-1,-2,-3]

# training loop
sess.run(tf.global_variables_initializer())
for i in range(1000):
    # if(sess.run(loss,{x:x_data,y:y_data})<1e-10):
    #     break
    sess.run(train,{x:x_data,y:y_data})

# evaluate training accuracy
now_W,now_b = sess.run(([W,b]))
now_loss = sess.run(loss,{x:x_data,y:y_data})
print("W: %s\nb: %s\nloss: %s"%(now_W,now_b,now_loss))
