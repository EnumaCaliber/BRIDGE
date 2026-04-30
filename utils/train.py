import torch
import numpy as np
import torch.nn.functional as F
import copy


def trainer_loader():
    return train





def initialize_weight(model, loader):
    batch = next(iter(loader))
    device = next(model.parameters()).device
    with torch.no_grad():
        model(batch[0].to(device))


def train(model, optpack, train_loader, test_loader, print_steps=100, log_results=False, log_path='log.txt',
          patience=30):
    model.train()
    opt = optpack["optimizer"](model.parameters())
    if optpack["scheduler"] is not None:
        sched = optpack["scheduler"](opt)
    else:
        sched = None
    num_steps = optpack["steps"]
    device = next(model.parameters()).device

    results_log = []
    training_step = 0
    epoch = 0

    # Best model tracking
    best_test_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    if sched is not None:
        while True:
            epoch += 1
            model.train()

            # Track training metrics for this epoch
            train_loss = 0
            correct = 0
            total = 0

            for i, (x, y) in enumerate(train_loader):
                training_step += 1
                x = x.to(device)
                y = y.to(device)

                opt.zero_grad()
                yhat = model(x)
                loss = F.cross_entropy(yhat, y)
                loss.backward()
                opt.step()
                sched.step()

                # Accumulate training metrics
                train_loss += loss.item()
                _, predicted = yhat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                if training_step >= num_steps:
                    break

            # Calculate epoch training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * correct / total

            # Evaluate on test set at the end of each epoch
            test_acc, test_loss = test(model, test_loader)

            # Track best model and print with indicator
            if test_acc > best_test_acc:
                print(
                    f'Epoch: {epoch} \t Steps: {training_step}/{num_steps} \t Train acc: {train_acc:.2f} \t Test acc: {test_acc:.2f} \t Train loss: {avg_train_loss:.4f} \t Test loss: {test_loss:.4f} \t ✓ NEW BEST (prev: {best_test_acc:.2f}%)')
                best_test_acc = test_acc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                print(
                    f'Epoch: {epoch} \t Steps: {training_step}/{num_steps} \t Train acc: {train_acc:.2f} \t Test acc: {test_acc:.2f} \t Train loss: {avg_train_loss:.4f} \t Test loss: {test_loss:.4f}')
                epochs_without_improvement += 1

            if log_results:
                results_log.append([test_acc, test_loss, train_acc, avg_train_loss])
                np.savetxt(log_path, results_log)

            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f'\nEarly stopping at epoch {epoch}: No improvement for {patience} epochs')
                break

            if training_step >= num_steps:
                break
    else:
        while True:
            epoch += 1
            model.train()

            # Track training metrics for this epoch
            train_loss = 0
            correct = 0
            total = 0

            for i, (x, y) in enumerate(train_loader):
                training_step += 1
                x = x.to(device)
                y = y.to(device)

                opt.zero_grad()
                yhat = model(x)
                loss = F.cross_entropy(yhat, y)
                loss.backward()
                opt.step()

                # Accumulate training metrics
                train_loss += loss.item()
                _, predicted = yhat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                if training_step >= num_steps:
                    break

            # Calculate epoch training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100. * correct / total

            # Evaluate on test set at the end of each epoch
            test_acc, test_loss = test(model, test_loader)

            # Track best model and print with indicator
            if test_acc > best_test_acc:
                print(
                    f'Epoch: {epoch} \t Steps: {training_step}/{num_steps} \t Train acc: {train_acc:.2f} \t Test acc: {test_acc:.2f} \t Train loss: {avg_train_loss:.4f} \t Test loss: {test_loss:.4f} \t ✓ NEW BEST (prev: {best_test_acc:.2f}%)')
                best_test_acc = test_acc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                print(
                    f'Epoch: {epoch} \t Steps: {training_step}/{num_steps} \t Train acc: {train_acc:.2f} \t Test acc: {test_acc:.2f} \t Train loss: {avg_train_loss:.4f} \t Test loss: {test_loss:.4f}')
                epochs_without_improvement += 1

            if log_results:
                results_log.append([test_acc, test_loss, train_acc, avg_train_loss])
                np.savetxt(log_path, results_log)

            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f'\nEarly stopping at epoch {epoch}: No improvement for {patience} epochs')
                break

            if training_step >= num_steps:
                break

    # Restore best model
    if best_model_state is not None:
        print(f'\nRestoring best model with test acc: {best_test_acc:.2f}%')
        model.load_state_dict(best_model_state)

    train_acc, train_loss = test(model, train_loader)
    test_acc, test_loss = test(model, test_loader)
    print(f'Train acc: {train_acc:.2f}\t Test acc: {test_acc:.2f}')
    return [test_acc, test_loss, train_acc, train_loss]


def test(model, loader):
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total_loss = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            yhat = model(x)
            loss = F.cross_entropy(yhat, y)

            total_loss += loss.item()
            _, pred = yhat.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / len(loader)

    model.train()

    return acc, avg_loss