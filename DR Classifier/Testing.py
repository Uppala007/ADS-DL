# Load the saved model
model = tf.keras.models.load_model('/content/diabetic_retinopathy_classifier.h5')

# Set the path to your test dataset
test_dir = os.path.join(base_dir, 'test')

# Prepare the test generator
test_generator = valid_test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
                                                        batch_size=batch_size, class_mode='binary', shuffle=False)

# Evaluate the model on the test data
test_generator.reset()
predictions = model.predict(test_generator, steps=test_generator.samples // batch_size + 1)
predicted_classes = np.where(predictions > 0.5, 1, 0).reshape(-1)
true_classes = test_generator.classes

# Classification report and Confusion Matrix
print(classification_report(true_classes, predicted_classes, target_names=test_generator.class_indices.keys()))
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
