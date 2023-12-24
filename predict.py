import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import json
import argparse
from train import load_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Image to predict')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint to use when predicting')
    parser.add_argument('--topk', type=int, help='Return top K predictions')
    parser.add_argument('--labels', type=str, help='JSON file containing label names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def load_image(image_path):
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    pil_image = Image.open(image_path)
    pil_image = img_loader(pil_image).float()

    image = np.array(pil_image)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np.transpose(image, (1, 2, 0)) - mean) / std
    image = np.transpose(image, (2, 0, 1))

    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)

    return image


def predict(image_path, checkpoint_path, topk=5, labels_path='', use_gpu=False):
    checkpoint_dict = torch.load(checkpoint_path)
    model_arch = checkpoint_dict['arch']
    num_labels = len(checkpoint_dict['class_to_idx'])
    hidden_units = checkpoint_dict['hidden_units']

    model = load_model(arch=model_arch, num_labels=num_labels, hidden_units=hidden_units)

    if use_gpu and torch.cuda.is_available():
        model.cuda()

    was_training = model.training
    model.eval()

    image = load_image(image_path)

    if use_gpu and torch.cuda.is_available():
        image = image.cuda()

    result = model(image).topk(topk)

    if use_gpu and torch.cuda.is_available():
        probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
        classes = result[1].data.cpu().numpy()[0]
    else:
        probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
        classes = result[1].data.numpy()[0]

    if labels_path:
        with open(labels_path, 'r') as f:
            cat_to_name = json.load(f)
        labels = list(cat_to_name.values())
        classes = [labels[x] for x in classes]

    model.train(mode=was_training)

    if image_path:
        print('Predictions and probabilities:', list(zip(classes, probs)))

    return probs, classes

if __name__ == "__main__":
    args = parse_arguments()

    if args.image and args.checkpoint:
        predict(args.image, args.checkpoint, topk=args.topk, labels_path=args.labels, use_gpu=args.gpu)
