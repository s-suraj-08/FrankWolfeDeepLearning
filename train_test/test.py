import torch

def testing(args,
           model,
           loader_test,
            device):
    # if args.EARLYSTOP:
    #     model.load_state_dict(torch.load("best_model"))

    model.eval()
    model.to(device)
    ################################################################
    #                           TESTING                            #
    ################################################################
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader_test:
            images = images.to(device)
            labels = labels.to(device)

            #out
            out = model(images)

            #predict
            _, prediction = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
        acc = 100 * correct / total
        print('Testing Accuracy of the network on the {} test images: {} %'.format(total,acc))
    return acc