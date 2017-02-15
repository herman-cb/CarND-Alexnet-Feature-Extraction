import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
with open('train.p', mode='rb') as f:
    train = pickle.load(f)

X_data = train['features']
y_data = train['labels']

# TODO: Split data into training and validation sets.
total_count = len(y_data)
print('Total examlples = {}'.format(total_count))
num_train = int(total_count * 0.8)

X_train = X_data[0:num_train]
y_train = y_data[0:num_train]

from sklearn.preprocessing import LabelBinarizer


X_validation = X_data[num_train:]
y_validation = y_data[num_train:]

print('Number of training examples={}'.format(num_train))
print('Number of validation examples={}'.format(total_count-num_train))

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
nb_classes = 43
fc8W = tf.Variable(tf.truncated_normal((fc7.get_shape().as_list()[-1], nb_classes)))
fc8b = tf.Variable(tf.zeros((nb_classes,)))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)



# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
y_one_hot = tf.one_hot(y, nb_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_one_hot)
loss_operation = tf.reduce_mean(cross_entropy)
lr = 0.001

optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

correct_prediction = tf.equal(tf.argmax(probs, 1), tf.argmax(y_one_hot, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

EPOCHS = 10
BATCH_SIZE = 128

from sklearn.utils import shuffle


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train)
    print('Training ...')

    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x:batch_x, y:batch_y})
        X_validation, y_validation = shuffle(X_validation, y_validation)
        validation_acc = evaluate(X_validation, y_validation)
        print('EPOCH = {} ...'.format(i+1))
        print('Validation accuracy = {}'.format(validation_acc))
        print()
    saver.save(sess, 'model')
    print('model saved')    

