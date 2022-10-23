import torch
import torchvision
import numpy as np
import cv2 as cv

import argparse
import os
import sys
sys.path.append(os.path.normpath( os.path.join(os.getcwd(), *([".."] * 1))))

from model.multiclass_lane_detection.regnet import Model
from utils import vis_lines
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
if torch.cuda.is_available():
    model.load_state_dict(torch.load('regnet_multiclass37.pth'))
else:
    model.load_state_dict(torch.load('regnet_multiclass37.pth', map_location='cpu'))
model.eval()


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required = False, help="Image path")
parser.add_argument('-f', '--film', required=False, help='Film path')
args = parser.parse_args()


def vis_film(film):
    cap = cv.VideoCapture(film)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv.resize(frame, (1280,720))
        im = transform(cv.cvtColor(frame, cv.COLOR_BGR2RGB)).unsqueeze(0).to(device)

        out = model(im).squeeze(0)
        out = out.max(dim=0)[1]
        output = out.detach().cpu().unsqueeze(0)
        output = torch.cat((output, output, output), dim=0)
        output = vis_lines(output).squeeze(0).byte()
        output = cv.cvtColor(output.numpy(), cv.COLOR_RGB2BGR)
        frame = cv.addWeighted(frame, 1, output, 0.5, 0.0)
        cv.imshow('frame', frame)
        cv.imshow('mask', output)
    
        if cv.waitKey(1) == ord('q'):
            break   
    cap.release()
    cv.destroyAllWindows()


def vis_img(img):
    print("Not implemented yet")


if __name__ == "__main__":
    
    if args.image != None:
        vis_img(args.image)
    
    elif args.film != None:
        vis_film(args.film)

