from earlystop import early_stop
from optim import get_optimizer,decay_optimizer
from dfw.losses import set_smoothing_enabled
import torch
import torch.nn as nn

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def training(args,
          loader_train,
         loader_val,
         model,
         loss,
         optim,
         smoothing,
         device,
         ):
    
    ################################################################
    model.to(device)
    if args.EARLYSTOP:
        earlystopping = early_stop()
    
    if(args.MODEL_TYPE != "cnn"):
        model.apply(weights_init)

    total_step = len(loader_train)
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    best_model = None    
    ################################################################
    #                           TRAINING                           #
    ################################################################
    for epoch in range(args.EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        for i, (images,labels) in enumerate(loader_train):
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            out = model(images)

            #loss
            if smoothing:
                set_smoothing_enabled(True)
                lossval = loss(out,labels)
            else:
                lossval = loss(out,labels)

            #chcking if L1 reg
            if args.TYPE == "L1":
                regloss = sum([p.abs().sum() for p in model.parameters()])
                lossval = lossval + regloss * args.REGPARAM
            
            #pred
            _, predicted = torch.max(out.data,1)
            correct += (predicted == labels).sum().item()

            #gradient backpass
            optim.zero_grad()
            lossval.backward()
            if args.OPT == 'dfw':
                optim.step(lambda: float(lossval))
            else:
                optim.step()

            total += labels.size(0)
            total_loss += lossval.sum().item()

#             if (i+1) % 200 == 0:
#                 print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                        .format(epoch+1, args.EPOCHS, i+1, total_step, lossval.item()))

        train_acc.append(100 * correct / total)
        train_loss.append(total_loss / total)

        if(((epoch+1) % 3 == 0) and(args.OPT == 'sgd')):
            #print("decay_factor:",args.DECAY_FACTOR)
            decay_optimizer(optim, args.DECAY_FACTOR)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for images, labels in loader_val:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                # Modified
                lossval = loss(outputs,labels)

                #predictions
                _, predicted = torch.max(outputs.data, 1)

                total_loss += lossval.sum().item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Modified
            val_acc.append(100 * correct / total)
            val_loss.append(total_loss / total)

            #print('Validataion accuracy is: {} %'.format(100 * correct / total))
            if args.EARLYSTOP:
                earlystopping(100 * correct / total)
                if earlystopping.accuracy_increased:
                    best_model = model
                    torch.save(best_model.state_dict(), "best_model") # Ideally it can be saved only once at the end

                if earlystopping.stop and best_model is not None:
                    break

        model.train()
    return train_acc, train_loss, val_acc, val_loss