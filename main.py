import tensorflow as tf

sess = tf.Session()


node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
sum = a + b
mul = sum * 3

W = tf.Variable([.5],dtype=tf.float32)
B = tf.Variable([-.5],dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# 测评数据达到目标的程度
# 差的平方之和 loos与lossv_2一样
model = W*x+B
loss = tf.reduce_sum(tf.square(model-y))
loosv_2 = tf.reduce_sum(tf.squared_difference(model,y))

sess.run(tf.global_variables_initializer())
print("sess.run(node3):", sess.run(sum,{a:[3,2,1],b:[1,2,3]}))
print(sess.run(mul, {a: 3, b: 4}))
# 重新分配W和B使得Model结果与y接近
# sess.run([tf.assign(W,[-1.]),tf.assign(B,[1.])])
print("model",sess.run(model,{x:[1,2,3,4]}))
print("loss",sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
print("loosv_2",sess.run(loosv_2,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
print(sess.run([W,B]))
print("loss",sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))