from torchvision import transforms as T

to_tensor = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

augmenter = T.Compose([
    # T.RandAugment(),
    to_tensor
])