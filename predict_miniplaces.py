from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from dataloader import MiniPlaces
from student_code import SimpleConvNet, train_model, test_model
import os
import random
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main(args):
    # set up random seed
    torch.manual_seed(0)

    ###################################
    # setup model, loss and optimizer #
    ###################################
    model = SimpleConvNet()

    training_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # set up transforms to transform the PIL Image to tensors
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    ################################
    # setup dataset and dataloader #
    ################################
    data_folder = './data'
    if not os.path.exists(data_folder):
        os.makedirs(os.path.expanduser(data_folder), exist_ok=True)

    train_set = MiniPlaces(
        root=data_folder, split="train", download=True, transform=train_transform)
    test_set = MiniPlaces(
        root=data_folder, split="val", download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False)
    

    
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # print images
    
    classes = ['abbey',
	'airport_terminal',
	'amphitheater',
	'amusement_park',
	'aquarium',
	'aqueduct',
	'art_gallery',
	'assembly_line',
	'auditorium',
	'badlands',
	'bakery/shop',
	'ballroom',
	'bamboo_forest',
	'banquet_hall',
	'bar',
	'baseball_field',
	'bathroom',
	'beauty_salon',
	'bedroom',
	'boat_deck',
	'bookstore',
	'botanical_garden',
	'bowling_alley',
	'boxing_ring',
	'bridge',
	'bus_interior',
	'butchers_shop',
	'campsite',
	'candy_store',
	'canyon',
	'cemetery',
	'chalet',
	'church/outdoor',
	'classroom',
	'clothing_store',
	'coast',
	'cockpit',
	'coffee_shop',
	'conference_room',
	'construction_site',
	'corn_field',
	'corridor',
	'courtyard',
	'dam',
	'desert/sand',
	'dining_room',
	'driveway',
	'fire_station',
	'food_court',
	'fountain',
	'gas_station',
	'golf_course',
	'harbor',
	'highway',
	'hospital_room',
	'hot_spring',
	'ice_skating_rink/outdoor',
	'iceberg',
	'kindergarden_classroom',
	'kitchen',
	'laundromat',
	'lighthouse',
	'living_room',
	'lobby',
	'locker_room',
	'market/outdoor',
	'martial_arts_gym',
	'monastery/outdoor',
	'mountain',
	'museum/indoor',
	'office',
	'palace',
	'parking_lot',
	'phone_booth',
	'playground',
	'racecourse',
	'railroad_track',
	'rainforest',
	'restaurant',
	'river',
	'rock_arch',
	'runway',
	'shed',
	'shower',
	'ski_slope',
	'skyscraper',
	'slum',
	'stadium/football',
	'stage/indoor',
	'staircase',
	'subway_station/platform',
	'supermarket',
	'swamp',
	'swimming_pool/outdoor',
	'temple/east_asia',
	'track/outdoor',
	'trench',
	'valley',
	'volcano',
	'yard']
    
    imshow(torchvision.utils.make_grid(images))
    print('Starting from top left: ', ' '.join('%5s' % classes[labels[j]] for j in range(32)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification using Pytorch')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='number of images within a mini-batch')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
    
    
   

