# import libraries

import os
import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

#############################################################################
'''
Here, the datasets we used in the paper can not be released for personal information protection.
Instead, you can identify a sample dataset.
please refer to "sample_dataset.csv"
'''

# Load dataset
data = pd.read_csv("your_own_dataset.csv") # or test with "sample_dataset.csv"
X_data = data.drop(["Outcome"], axis=1)
y_data = data["Outcome"]

# Input variable vectorization (one hot encoding)
data_type_info = pd.read_csv("your_own_data_type_file.csv") # or test with "data_type_info.csv"

int_cols = data_type_info.query('type == "integer"')['variable']
float_cols = data_type_info.query('type == "float"')['variable']
binary_cat_cols = data_type_info.query('type == "binary_categorical"')['variable']
other_cat_cols = data_type_info.query('type == "categorical"')['variable']

def type_conv(data, int_cols, float_cols, binary_cat_cols, other_cat_cols):

    data.loc[:, int_cols] = data.loc[:, int_cols].astype("int32")
    data.loc[:, float_cols] = data.loc[:, float_cols].astype("float32")

    for col in binary_cat_cols:
        data[col] = data[col].astype("category").cat.codes

    data.loc[:, other_cat_cols] = data.loc[:, other_cat_cols].astype("object")

    data = pd.get_dummies(data)

    return data

X_data = type_conv(X_data, int_cols, float_cols, binary_cat_cols, other_cat_cols)

# Split data into training & model tuning and test datasets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_data, y_data,
    test_size=0.1,
    stratify=y_data,
    random_state=1004
)

# Normalize features
numeric_vars = list(int_cols) + list(float_cols)

X_trainval_scaled = X_trainval.copy()
X_test_scaled = X_test.copy()

scaler = StandardScaler()
scaler.fit(X_trainval.loc[:, numeric_vars])

X_trainval_scaled.loc[:, numeric_vars] = scaler.transform(X_trainval_scaled.loc[:, numeric_vars])
X_test_scaled.loc[:, numeric_vars] = scaler.transform(X_test_scaled.loc[:, numeric_vars])

# Split training & model tuning datasets into training and model tuning datasets
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
    X_trainval_scaled, y_trainval,
    test_size=1/9,
    stratify=y_trainval,
    random_state=1004
)

# Convert pandas dataframe to numpy array
X_trainval_scaled = X_trainval_scaled.values
X_val_scaled = X_val_scaled.values
X_test_scaled = X_test_scaled.values


#############################################################################
# Distributed computation (GPU parallel computing)
mirrored_strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

BUFFER_SIZE = len(X_train_scaled)
BATCH_SIZE_PER_REPLICA = 32
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

train_datasets = tf.data.Dataset.from_tensor_slices((X_trainval_scaled, y_trainval)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
valid_datasets = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val)).batch(BATCH_SIZE)
test_datasets = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test)).batch(BATCH_SIZE)

# set params for gridesearch (tensorboard)
variable_num = X_data.shape[1]
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([3, 4, 5]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([variable_num*2, variable_num*3, variable_num*4]))
HP_DROPOUT = hp.HParam('dropout_rate', hp.Discrete([0.1, 0.2]))

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_LAYERS, HP_NUM_UNITS, HP_DROPOUT],
        metrics=[hp.Metric('accuracy', display_name='Accuracy'),
                 hp.Metric('auc', display_name='AUROC')]
  )

# build a model
def train_model(train_datasets, hparams, log_dir, valid_datasets=None, learning_rate=0.001, batchnorm=True, dropout=True, final=False):

    tf.keras.backend.clear_session()

    #----- model structure
    with mirrored_strategy.scope():
        inputs = tf.keras.layers.Input(shape=(118,))
        x = inputs

        for i in range(hparams[HP_NUM_LAYERS]):
            x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS],
                                      activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
            if batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)
            if dropout:
                x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)

        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        #----- model compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

    #----- model train
    ckpt_path = os.path.join(os.getcwd(), 'temp.h5')
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.2,
                                                     verbose = 1,
                                                     patience=5,
                                                     min_lr=1e-15)

    model.fit(train_datasets,
              validation_data=valid_datasets if final is False else None,
              epochs=100,
              callbacks=[ckpt, tensorboard_cb, reduce_lr] if final is False else [tensorboard_cb, reduce_lr],
              verbose=1)

    if not valid_datasets is None:
        model.load_weights(ckpt_path)

    return model

# build model training process
def run(train_datasets, hparams, valid_datasets, run_name):

    trained_model = train_model(train_datasets, hparams,
                                log_dir=os.path.join('logs/hparam_tuning/', run_name),
                                valid_datasets=valid_datasets)

    _, accuracy, _ = trained_model.evaluate(valid_datasets)

    y_pred = trained_model.predict(X_val_scaled)
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)

    with tf.summary.create_file_writer(os.path.join('logs/hparam_tuning_results/', run_name)).as_default():
        hp.hparams(hparams, trial_id=run_name)
        tf.summary.scalar('accuracy', accuracy, step=1)
        tf.summary.scalar('auc', roc_auc, step=1)

    return accuracy, roc_auc

# grid search
s_results = {}
session_num = 0

for num_layers in HP_NUM_LAYERS.domain.values:
    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in HP_DROPOUT.domain.values:
            hparams = {
                HP_NUM_LAYERS: num_layers,
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate
                }
            run_name = "run-%d" % session_num

            print()
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})

            accuracy, roc_auc = run(train_datasets, hparams, valid_datasets, run_name)

            s_results[run_name] = {'hparams': hparams,
                                   'accuracy': accuracy,
                                   'roc_auc': roc_auc}
            session_num += 1

# print best params
df_results = pd.DataFrame(s_results).T
best_param = df_results.sort_values(by='mean_accuracy', ascending=False).head(1)['hparams']
print('the best params:\n', {h.name: best_param.item()[h] for h in best_param.item()})


#############################################################################
# model retrain with whole trainval datasets and best params
final_model = train_model(train_datasets, best_param.item(), log_dir='./logs/final_model', final=True)
final_model.save('./final_model.h5') # save the final model

# load the saved model and calculate auroc
saved_model = tf.keras.models.load_model('./final_model.h5')
fpr, tpr, thresholds = roc_curve(y_test, saved_model.predict(X_test_scaled))
roc_auc = auc(fpr, tpr)
print(roc_auc)