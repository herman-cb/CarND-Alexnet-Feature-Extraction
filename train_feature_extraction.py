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

X_validation = X_data[num_train:]
y_validation = y_data[num_train:]

print('Number of training examples={}'.format(num_train))
print('Number of validation examples={}'.format(total_count-num_train))

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
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

# TODO: Train and evaluate the feature extraction model.

# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
