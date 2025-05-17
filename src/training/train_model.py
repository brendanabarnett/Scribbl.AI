import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from src.config import MODEL_PATH, SCALER_PATH

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={'0': 'label'}, inplace=True)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y

def train(csv_path, epochs=8, batch_size=256):
    X, y = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    num_classes = y_train.shape[1]
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Dropout(0.2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    model.save(MODEL_PATH)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    train(args.data, epochs=args.epochs, batch_size=args.batch_size)
