import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

train = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
label = tf.constant([1, 3, 5, 7, 9], dtype=tf.float32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

history = model.fit(train, label, verbose=0, epochs=1000)

y_pred = model.predict([6.0], verbose=2)

model.save('model.h5')