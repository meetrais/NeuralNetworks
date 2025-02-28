# Import necessary libraries
import torch
import torchvision.models as models
import torchvision.transforms as T
import requests
from PIL import Image
from io import BytesIO

def download_image(url):
    """
    Download an image from the given URL with a valid
    User-Agent and return it as a PIL Image.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises an HTTPError if the response is unsuccessful
    return Image.open(BytesIO(response.content)).convert('RGB')

def load_imagenet_labels():
    """
    Download the 1,000 ImageNet class labels with a valid
    User-Agent and return them as a list indexed by class ID.
    """
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    
    response = requests.get(labels_url, headers=headers)
    response.raise_for_status()
    labels_str = response.text.strip().split("\n")
    return labels_str

def preprocess_image(image):
    """
    Transform the PIL Image into a Tensor suitable for
    the pretrained PyTorch model.
    """
    transform = T.Compose([
        T.Resize(256),             # Resize the image (shortest side = 256)
        T.CenterCrop(224),         # Crop the image at the center to 224x224
        T.ToTensor(),              # Convert the image to a PyTorch tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])  # Normalize using ImageNet means and stds
    ])
    return transform(image).unsqueeze(0)  # Add a batch dimension

def classify_image_with_resnet(image_url, topk=5):
    """
    Download an image, preprocess it, run it through
    ResNet-50, and print the top-k predictions.
    """
    # Download and preprocess
    pil_img = download_image(image_url)
    img_t = preprocess_image(pil_img)

    # Load a pretrained ResNet-50 model
    model = models.resnet50(pretrained=True)
    model.eval()  # Set model to evaluation mode

    # Disable gradient calculation for speed and memory
    with torch.no_grad():
        output = model(img_t)

    # Get label names and calculate probabilities
    labels = load_imagenet_labels()
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top-k probabilities and corresponding indices
    topk_probs, topk_indices = torch.topk(probabilities, topk)

    # Print the results
    print(f"\nImage URL: {image_url}")
    print("Top Predictions:")
    for i in range(topk):
        class_idx = topk_indices[i].item()
        prob = topk_probs[i].item()
        print(f"  {labels[class_idx]:<30} ({prob:.4f})")

if __name__ == "__main__":
    # Replace this URL with any image URL you want to test
    test_image_url = (
        "https://images.pexels.com/photos/30926559/pexels-photo-30926559/free-photo-of-close-up-of-housefly-on-red-surface.jpeg?auto=compress&cs=tinysrgb&w=800"
    )

    # Classify the image and print top-5 predictions
    classify_image_with_resnet(test_image_url, topk=5)