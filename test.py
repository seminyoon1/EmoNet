import torch
from torchvision.transforms import transforms
from models.vgg16 import Classifier
from data_loader import EmotionNet

# Load the saved model
model = Classifier()
model.load_state_dict(torch.load('snapshot\\models\\EmotionNet\\normal\\fold_0\\Imagenet\\03_43.pth'))
model.eval()

# Set the required parameters
image_size = 256
metadata_path = './data/dataset/Dog Emotion'
mode = 'test'
fold = 0

# Define the transformation with resizing
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Apply appropriate transformations
])

# Create the test dataset
test_dataset = EmotionNet(image_size=image_size, metadata_path=metadata_path, transform=transform, mode=mode, fold=fold)

# Create DataLoader for batch processing
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a function for evaluation
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Evaluate the model
test_accuracy = evaluate_model(model, test_loader)
print('Test Accuracy: {:.2f}%'.format(test_accuracy))