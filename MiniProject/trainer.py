import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tuner import RandomSearch

# Load the dataset
data = pd.read_csv('insurance.csv')

# Preprocess the data
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])  # Male=1, Female=0
data['smoker'] = le.fit_transform(data['smoker'])  # Yes=1, No=0
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Features and target
X = data.drop(['charges'], axis=1)
y = data['charges']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Configure GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) < 7:
    print(f"Only {len(gpus)} GPUs available, but proceeding with available GPUs.")
else:
    tf.config.experimental.set_visible_devices(gpus[:7], 'GPU')

# Function to build the model
def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units_layer1', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_layer1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(hp.Int('units_layer2', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_layer2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
best_fold = None
best_mse = float('inf')
best_model_file = None

for train_index, val_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Hyperparameter tuning
    tuner = RandomSearch(
        build_model,
        objective='mse',
        max_trials=5,
        directory='hyperparameter_tuning',
        project_name=f'fold_{fold}'
    )
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=1)

    # Best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build and train the final model with the best hyperparameters
    best_model = tuner.hypermodel.build(best_hps)

    # Save weights during training
    model_file = f'model_fold_{fold}.keras'
    checkpoint = ModelCheckpoint(model_file, save_best_only=True, monitor='val_loss', mode='min')

    # Train the model
    best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[checkpoint])

    # Evaluate the model
    val_predictions = best_model.predict(X_val)
    mse = mean_squared_error(y_val, val_predictions)
    print(f"Fold {fold} MSE: {mse}")

    # Update best model
    if mse < best_mse:
        best_mse = mse
        best_fold = fold
        best_model_file = model_file

    fold += 1

print(f"Best Model is from Fold {best_fold} with MSE: {best_mse}")
print(f"Best Model saved as: {best_model_file}")
