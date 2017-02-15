import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

EPOCHS = 10
BATCH_SIZE = 128
nb_classes = 43

with open('train.p', mode='rb') as f:
    data = pickle.load(f)

X_train, X_validation, y_train, y_validation = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)

print('Number of training examples={}'.format(X_train.shape[0]))
print('Number of validation examples={}'.format(X_validation.shape[0]))

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(x, (227, 227))

fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

fc8W = tf.Variable(tf.truncated_normal((fc7.get_shape().as_list()[-1], nb_classes), stddev=1e-2))
fc8b = tf.Variable(tf.zeros((nb_classes)))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
loss_operation = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

preds = tf.argmax(logits, 1)
correct_prediction = tf.equal(preds, y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data, sess):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(loss_operation, feed_dict={x: batch_x, y:batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_accuracy / num_examples, total_loss / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = X_train.shape[0]
    print('Training ...')
    t0 = time.time()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)

        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x:batch_x, y:batch_y})

        val_acc, val_loss = evaluate(X_validation, y_validation, sess)
        print('time = {} seconds'.format(time.time()-t0))
        print('EPOCH = {} ...'.format(i+1))
        print('Validation accuracy = {}, validation loss = {}'.format(val_acc, val_loss))
    saver.save(sess, 'model')
    print('model saved')    

