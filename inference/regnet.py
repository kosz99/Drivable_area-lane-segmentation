import torch
import torchvision
import numpy as np
import cv2 as cv

import argparse
import os
import sys
sys.path.append(os.path.normpath( os.path.join(os.getcwd(), *([".."] * 1))))

from model.lane_detection.regnet import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
if torch.cuda.is_available():
    model.load_state_dict(torch.load('regnet.pth'))
else:
    model.load_state_dict(torch.load('regnet.pth', map_location='cpu'))
    
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

        out = model(im).squeeze(0)*255.0 
        out = out.repeat(3,1,1).permute(1,2,0).detach().type(torch.uint8).cpu().numpy()
        out[:,:,0:2] = 0

        frame = cv.addWeighted(frame, 1, out, 0.5, 0.0)
        cv.imshow('frame', frame)
        cv.imshow('mask', out)
    
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

