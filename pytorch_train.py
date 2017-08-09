import torch
from torch.autograd import Variable

from PytorchModels.base_model import Model
from PytorchModels.xnor_model import XNORModel
from datasets.pytorch_provider import get_loaders


# Initialize model
cfg = {
    'epochs': 10,
    'lr': 0.001,
    'momentum': 0.9,
    'batch_size': 4,
    'report_step': 2000,
    'model_type': 'xnor'  # choices from ['base', 'xnor']
}

train_loader, test_loader = get_loaders(batch_size=cfg['batch_size'])
if cfg['model_type'] == 'base':
    net = Model()
elif cfg['model_type'] == 'xnor':
    net = XNORModel()
else:
    raise NotImplementedError()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(),
    lr=cfg['lr'],
    momentum=cfg['momentum'])


# Train the model
for epoch in range(cfg['epochs']):

    running_loss = 0.0
    for batch_n, (inputs, labels) in enumerate(train_loader):
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if batch_n % cfg['report_step'] == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_n + 1, running_loss / cfg['report_step']))
            running_loss = 0.0

    # Test the model
    correct = 0
    total = 0
    for (images, labels) in test_loader:
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
