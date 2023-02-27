import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Load the data
train_data = pd.read_csv("merged_data_train.csv")
test_data = pd.read_csv("merged_data_test.csv")

# Define the text columns
text_columns = ['industry', 'location']

# Define the target column
target_column = 'moved_after_2019'

# Split the data into features and target
X = train_data.drop(target_column, axis=1)
y = train_data[target_column]

# Flatten the input data
X_flat = X[text_columns].values.flatten()

# Create a TextVectorization layer
vectorizer = TextVectorization(output_mode="int")

# Adapt the vectorizer to the training data
vectorizer.adapt(X_flat)

# Convert the text data to sequences of integers
X_seq = vectorizer(X_flat)

# Reshape the sequences to match the original shape
X_seq = tf.reshape(X_seq, [-1, len(text_columns)])

# Define the deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_dim=X_seq.shape[1]),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_seq, y, epochs=10)

# Make predictions on the test data
test_data_flat = test_data[text_columns].values.flatten()
test_data_seq = vectorizer(test_data_flat)
test_data_seq = tf.reshape(test_data_seq, [-1, len(text_columns)])
y_pred = model.predict_classes(test_data_seq).flatten()

# Save the predicted results as a CSV file
results_df = pd.DataFrame({'user_id': test_data['user_id'], 'moved_after_2019': y_pred})
results_df.to_csv('submission.csv', index=False)
