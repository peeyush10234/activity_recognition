import tensorflow as tf
import model
import data_pre_processing
import numpy as np
import sklearn.metrics
import one_cycle_policy

if __name__ == '__main__':
    data_dir = 'data/'
    data_obj = data_pre_processing.DataPreprocessing()
    X_train, y_train, X_val, y_val = data_obj.perform_pre_processing()
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = model.CnnLstm(input_shape)

    epochs = 10
    batch_size = 64

    optimizer = tf.keras.optimizers.Adam( learning_rate=0.001, beta_1=0.9,
                                          beta_2=0.999, epsilon=1e-07,
                                          amsgrad=False,name='Adam')

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"]
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data = (X_val, y_val),
    )

    y_pred = model.predict(X_val)
    y_pred = np.argmax(y_pred, axis=1)
    y_val = np.argmax(y_val, axis=1)

    acc_score = sklearn.metrics.accuracy_score(y_val, y_pred)
    f1_score = sklearn.metrics.f1_score(y_val, y_pred, average='macro')
    recall = sklearn.metrics.recall_score(y_val, y_pred, average='macro')
    precision = sklearn.metrics.precision_score(y_val, y_pred, average='macro')

    print(f'accuracy --> {acc_score}',
          f'f1_score --> {f1_score}',
          f'recall --> {recall}',
          f'precision --> {precision}')



