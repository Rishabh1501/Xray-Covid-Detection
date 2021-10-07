from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import copy, os, time

#check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class Training():
    
    def __init__(self,batch_size=20,epochs=100,num_workers=0,
                 learning_rate=0.0001,momentum=0.9,
                 dataset_path='./data/',model_save_folder='models'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model_save_folder = model_save_folder
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_path, x),
                                          data_transforms[x])
                          for x in ['train', 'val']}
        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= batch_size,
                                                           shuffle=True, num_workers=num_workers)
                       for x in ['train', 'val']}
        
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.class_names = image_datasets['train'].classes  ## 0: child, and 1: nonchild
    
    
    def train_model(self):
        since = time.time()
        
        model = torchvision.models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default   
        num_ftrs = model.fc.in_features        
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device) 

         
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.fc.parameters(), lr= self.learning_rate, momentum= self.momentum)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        train_acc= list()
        valid_acc= list()

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch+1, self.epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                cur_batch_ind= 0
                for inputs, labels in self.dataloaders[phase]:
                    #print(cur_batch_ind,"batch inputs shape:", inputs.shape)
                    #print(cur_batch_ind,"batch label shape:", labels.shape)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    cur_acc= torch.sum(preds == labels.data).double()/self.batch_size
                    cur_batch_ind +=1
                    print("\npreds:", preds)
                    print("label:", labels.data)
                    print("%d-th epoch, %d-th batch (size=%d), %s acc= %.3f \n" %(epoch+1, cur_batch_ind, len(labels), phase, cur_acc ))
                    
                    if phase=='train':
                        train_acc.append(cur_acc)
                    else:
                        valid_acc.append(cur_acc)
                    
                epoch_loss= running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                print('{} Loss: {:.4f} Acc: {:.4f} \n\n'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch= epoch
                    best_model_wts = copy.deepcopy(model.state_dict())



        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc= %.3f at Epoch: %d' %(best_acc,best_epoch) )

        # load best model weights
        model.load_state_dict(best_model_wts)
        model.eval()
        save_path = os.path.join(self.model_save_folder,
                                 './covid_resnet18_epoch%d.pt' %self.epochs)
        torch.save(model,save_path)
        return train_acc, valid_acc