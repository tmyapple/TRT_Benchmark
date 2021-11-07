from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet')
# Save the entire model as a SavedModel.
model.save('resnet50_saved_model')