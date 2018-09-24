import torch

from torch.optim import SGD, Adam, Adadelta, Adagrad, Adamax, ASGD, LBFGS, RMSprop, Rprop
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, Resize

from utils.dataset import VOCTrain
from utils.criterion import CrossEntropyLoss2d
from utils.transform import Relabel, ToLabel, Colorize

def save_model(model_name, epoch, step):
    filename = f'{model_name}-{epoch:03}-{step:04}.pth'
    torch.save(model.state_dict(), filename)
    print(f'save: {filename} (epoch: {epoch}, step: {step})')

NUM_CLASSES = 22                # For Pascal VOC 2012, NUM_CLASSES shoud be equals to '22'.
IMG_SIZE    = (256,256)         # All images will be resized to 256x256 pixels.
NUM_EPOCHS  = 15                # Number of epochs
STEPS_LOSS  = 1                 # default:50 (0 = DISABLED)
STEPS_SAVE  = 0                 # default:500 (0 = DISABLED)
BATCH_SIZE  = 1                 # Num of training images per step
NUM_WORKERS = 1                 # Num of workers
MODEL_NAME   = 'trained-models/state'   # Name of the model to save
DATASET_PATH = 'datasets/VOC2012'       # Path of the dataset

cuda_enabled = torch.cuda.is_available()
print("CUDA_ENABLED: ", cuda_enabled)

input_transform = Compose([
    CenterCrop(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform = Compose([
    CenterCrop(256),
    ToLabel(),
    Relabel(255, 21),
])

# SegNet, FCN8, FCN16, FCN32, PSPNet, UNet
from networks.SegNet import *
Net = SegNet

model = Net(NUM_CLASSES)
if cuda_enabled:
    model = model.cuda()

model.train()

loader = DataLoader(
    VOCTrain(DATASET_PATH, input_transform, target_transform), 
    num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True
)

criterion = CrossEntropyLoss2d()

#optimizer = Adam(model.parameters(), lr=1e-3) # default Adam
optimizer = SGD(model.parameters(), lr=.1, momentum=.9) # default SGD
#optimizer = SGD(model.parameters(), lr=1e-3, momentum=.9) # original SGD
#optimizer = Adadelta(model.parameters()) # default Adadelta
#optimizer = Adagrad(model.parameters()) # default Adagrad
#optimizer = Adamax(model.parameters()) # default Adamax
#optimizer = ASGD(model.parameters()) # default ASGD
#optimizer = LBFGS(model.parameters()) # default LBFGS
#optimizer = RMSprop(model.parameters()) # default RMSprop
#optimizer = Rprop(model.parameters()) # default Rprop

iteration = 1
for epoch in range(1, NUM_EPOCHS+1):
    epoch_loss = []
    
    for step, (images, labels) in enumerate(loader):
        if cuda_enabled:
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        targets = Variable(labels)
        outputs = model(inputs)

        optimizer.zero_grad()

        loss = criterion(outputs, targets[:, 0])
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data)
        average = sum(epoch_loss) / len(epoch_loss)

        if STEPS_LOSS > 0 and step % STEPS_LOSS == 0:
            print(f'loss: {average} (epoch: {epoch}, step: {step})')

        if STEPS_SAVE > 0 and step % STEPS_SAVE == 0:
            save_model(MODEL_NAME, epoch, step)

        iteration = iteration + 1
        #break
    #break

save_model(MODEL_NAME, epoch, step)
print('Finished!')