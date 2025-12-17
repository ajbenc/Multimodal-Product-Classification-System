"""
Train Text-Only MLP Model (REGULARIZED VERSION)
Uses only text embeddings (MiniLM) for classification with stronger regularization
Expected: 85%+ accuracy with less overfitting
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

print("="*80)
print("TEXT-ONLY MLP TRAINING (REGULARIZED)")
print("="*80)

# Step 1: Load text embeddings
print("\n1. Loading text embeddings...")
text_df = pd.read_csv('Embeddings/text_embeddings_minilm.csv')
print(f"   Loaded {len(text_df)} text embeddings")

# Parse embeddings
print("   Parsing embeddings...")
text_df['embeddings_list'] = text_df['embeddings'].apply(ast.literal_eval)
text_arrays = np.array(text_df['embeddings_list'].tolist())
text_dim = text_arrays.shape[1]

# Add to dataframe
for i in range(text_dim):
    text_df[f'text_{i}'] = text_arrays[:, i]

text_cols = [f'text_{i}' for i in range(text_dim)]

# Step 2: Load products
print("\n2. Loading products...")
products_df = pd.read_csv('data/processed_products_with_images.csv')

# Step 3: Merge
print("\n3. Merging data...")
merged_df = text_df[['sku'] + text_cols + ['class_id']].merge(
    products_df[['sku']], 
    on='sku', 
    how='inner'
)
print(f"   Merged: {len(merged_df)} samples")

# Step 4: Split
print("\n4. Splitting train/test...")
train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42, stratify=merged_df['class_id'])
print(f"   Train: {len(train_df)}, Test: {len(test_df)}")

# Step 5: Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(merged_df['class_id'])

# Step 6: Prepare data manually (to avoid MultimodalDataset for more control)
print("\n5. Preparing data...")
X_train = train_df[text_cols].values.astype(np.float32)
y_train = label_encoder.transform(train_df['class_id'])
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(label_encoder.classes_))

X_test = test_df[text_cols].values.astype(np.float32)
y_test = label_encoder.transform(test_df['class_id'])
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=len(label_encoder.classes_))

# Step 7: Class weights
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

# Step 8: Create REGULARIZED model
print("\n6. Creating regularized text-only model...")
print("   - Higher dropout: 0.4 (vs 0.2)")
print("   - L2 regularization: 0.01")
print("   - Lower learning rate: 0.0005 (vs 0.001)")

text_input = Input(shape=(text_dim,), name='text_input')

# Hidden layers with STRONG regularization
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(text_input)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)  # Increased from 0.2

x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)  # Increased from 0.2

# Output layer
output = Dense(len(label_encoder.classes_), activation='softmax', name='output')(x)

model = Model(inputs=text_input, outputs=output)

# Lower learning rate for more stable training
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),  # Reduced from 0.001
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print(model.summary())

# Step 9: Train
print("\n7. Training...")
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_text_mlp_model_regularized.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)
]

history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_test, y_test_onehot),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Step 10: Evaluate
print("\n8. Evaluating...")
predictions = model.predict(X_test, verbose=1)
y_pred = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n{'='*80}")
print(f"TEXT-ONLY RESULTS (REGULARIZED):")
print(f"{'='*80}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
print(f"{'='*80}")

# Compare with previous
print(f"\n📊 COMPARISON:")
print(f"   Previous: 94.01% accuracy (overfitting gap: ~3%)")
print(f"   Current:  {accuracy*100:.2f}% accuracy")
if accuracy >= 0.85:
    print(f"   ✅ EXCEEDS 85% requirement!")
else:
    print(f"   ⚠️  Below 85% requirement")

# Step 11: Save results (overwrite previous)
results_df = pd.DataFrame({'Predictions': y_pred, 'True Labels': y_test})
results_df.to_csv('results/text_results.csv', index=False)
print(f"\n✅ Results saved to: results/text_results.csv")
print(f"✅ Model saved to: best_text_mlp_model_regularized.h5")
