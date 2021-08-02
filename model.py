import tensorflow as tf

class CnnLstm(tf.keras.Model):

    def __init__(self, input_shape, output_shape):
        super(CnnLstm, self).__init__()
        self.input = tf.keras.layers.Input(shape=input_shape)
        self.conv2D = tf.keras.layers.Conv2D(128, kernel_size=(5,1), activation=tf.nn.relu)
        self.reshape = tf.keras.layers.Reshape(24, 6*128)
        self.lstm1 = tf.keras.layers.LSTM(128, activation=tf.nn.relu, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(256, activation=tf.nn.tanh)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(output_shape, activation=tf.nn.softmax)

    def __call__(self):
        x = self.conv2D(self.input)
        for idx in range(3):
            x = self.conv2D(x)
        x = self.reshape(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        output = self.dense(x)
        return output











