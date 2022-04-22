import torch
import numpy as np

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device) # 주어진 수 내 랜덤하게 자연수 생성
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


from torch.autograd import Variable

# number of epochs to train the model
n_epochs = 100

valid_loss_min = np.inf  # track change in validation loss 100epoch

# keep track of training and validation loss
valid_loss = torch.zeros(n_epochs)
valid_F1 = torch.zeros(n_epochs)

model.to(device)

for e in range(0, n_epochs):

    ###################
    # train the model #
    ###################
    model.train()
    for inputs, targets in tqdm(train_dataloader):
        x, y = inputs.to(device), targets.to(device)
        # move tensors to GPU if CUDA is available
        x, targets_a, targets_b, lam = mixup_data(x, y, 1)

        outputs = model(x)

        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        optimizer.zero_grad()
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

    ######################
    # validate the model #
    ######################
    with torch.no_grad():
        model.eval()
        for inputs, targets in tqdm(valid_dataloader):
            # move tensors to GPU if CUDA is available
            inputs, targets = inputs.to(device), targets.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            logits = model(inputs)
            # calculate the batch loss
            loss = criterion(logits, targets)
            # update average validation loss
            valid_loss[e] += loss.item()
            # update training score
            logits = logits.argmax(1).detach().cpu().numpy().tolist()
            targets = targets.detach().cpu().numpy().tolist()
            valid_F1[e] += score_function(targets, logits)

    # calculate average losses
    valid_loss[e] /= len(valid_dataloader)
    valid_F1[e] /= len(valid_dataloader)

    scheduler.step(valid_loss[e])
    # print training/validation statistics
    print('Epoch: {} \tValidation Loss: {:.6f}'.format(
        e, valid_loss[e]))

    # print training/validation statistics
    print('Epoch: {} \tValidation accuracy: {:.6f}'.format(
        e, valid_F1[e]))

    # save model if validation loss has decreased
    if valid_loss[e] <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss[e]))
        torch.save(model.state_dict(), 'b0_mixup.pt')
        valid_loss_min = valid_loss[e]