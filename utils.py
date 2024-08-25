import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings to keep the output clean
import matplotlib.pyplot as plt
import os
import json

from drops_methods.resnet_concrete import ConcreteDropout  # Import ConcreteDropout from a custom module

def print_bounded_title(drop_method, dropout_rate):
    """
    Prints a title surrounded by a border of asterisks.

    Args:
        drop_method (str): Dropout method being used.
        dropout_rate (float or tuple): Dropout rate or rates for the method.
    """
    title = f"* Dropout method: {drop_method}  |  dropout rate: {dropout_rate} *"
    length = len(title)
    print('*' * length)
    print(title)
    print('*' * length)

class Hyper_Params:
    def __init__(self, params_dict=None):
        """
        Initialize hyperparameters with a dictionary.

        Args:
            params_dict (dict, optional): Dictionary of hyperparameters.
        """
        if params_dict:
            self.params_dict = params_dict
        else:
            # Default parameters
            self.params_dict = {
                'batch_size': None,
                'model_params': {},
                'epochs': None,
                'lr': None,
                'momentum': None,
                'optimizer': None,
                'scheduler_step_size': None,
                'scheduler_gamma': None,
                'epoch_loss_train': [],
                'epoch_accuracy_train': [],
                'train_loss_record': [],
                'train_accuracy_record': [],
                'test_loss_record': [],
                'test_accuracy_record': [],
            }
            
    def __getattr__(self, attr):
        """
        Get an attribute from params_dict.

        Args:
            attr (str): Attribute name.
        """
        if attr in self.params_dict:
            return self.params_dict[attr]
        else:
            raise AttributeError(f"'Params' object has no attribute '{attr}'")
        
    def __setattr__(self, attr, value):
        """
        Set an attribute in params_dict.

        Args:
            attr (str): Attribute name.
            value: Attribute value.
        """
        if attr == 'params_dict':
            super().__setattr__(attr, value)
        else:
            self.params_dict[attr] = value
            
    def __getitem__(self, key):
        """
        Get an item from params_dict.

        Args:
            key (str): Key name.
        """
        return self.params_dict[key]

def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion, acc_function, drop_method):
    """
    Evaluate the model's performance on the test dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion: Loss function.
        acc_function: Function to compute accuracy.
        drop_method (str): Dropout method used in the model.

    Returns:
        Tuple of average test loss and accuracy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Test_accuracy = []
    Test_loss = []

    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            data, target = batch
            
            if drop_method == 'Variational':
                output, kld_loss = model(data.to(device))
                loss_r = criterion(output, target.to(device))
                loss = loss_r + kld_loss
            else:
                output = model(data.to(device))
                target = target.reshape(target.shape[0])
                loss = criterion(output, target.to(device))
            
            accuracy = acc_function(output, target.to(device))
            
            Test_accuracy.append(accuracy)
            Test_loss.append(loss.item())
    
    return np.mean(Test_loss), np.mean(Test_accuracy)

def calculate_mean_std(loader: DataLoader):
    """
    Calculate the mean and standard deviation of the dataset.

    Args:
        loader (DataLoader): DataLoader for the dataset.

    Returns:
        Tuple of mean and standard deviation.
    """
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # Batch size
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean, std

def calculate_accuracy(model, dataloader, device):
    """
    Calculate the accuracy and confusion matrix of the model on the dataset.

    Args:
        model: The model to be evaluated.
        dataloader: DataLoader for the dataset.
        device: Device to run the model on ('cuda' or 'cpu').

    Returns:
        Tuple of accuracy and confusion matrix.
    """
    model.eval()
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([7, 7], int)

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1
    
    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix




def train_and_evaluate(model_class, dropout_values, train_loader, test_loader, criterion, hparams, drop_method):
    """
    Train and evaluate a model with different dropout values and methods.

    Args:
        model_class: Class of the model to be trained.
        dropout_values: List of dropout rates to be tested.
        train_loader: DataLoader for the training data.
        test_loader: DataLoader for the testing data.
        criterion: Loss function.
        hparams: Hyperparameters including learning rate, momentum, etc.
        drop_method: Type of dropout method used ('DropShake', 'Concrete', etc.).

    Returns:
        best_test_accuracies: Best test accuracies for the best dropout rate.
        best_droprate: Best dropout rate based on test accuracy.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    best_acc = 0
    best_droprate = (0.0, 0.0)  # Default value for dropout rates
    best_test_accuracies = []

    for dropout_rate in dropout_values:
        # Convert list of dropout ranges to a tuple to use as a dictionary key
        if drop_method == 'ShakeDrop':
            dropout_rate = tuple(dropout_rate)
        print_bounded_title(drop_method, dropout_rate)
        
        # Initialize model with the current dropout rate
        model = model_class(dropout_rate=dropout_rate).to(device)

        # Handle special dropout method configurations
        if drop_method == 'Concrete':
            N = len(train_loader.dataset)  # Number of training samples
            l = dropout_rate[0]  # Alpha value
            wr = l**2. / N  # Weight regularization
            dr = 2. / N  # Dropout regularization
            model = model_class(dropout_rate=0.0, num_classes=7, weight_regulariser=wr, dropout_regulariser=dr).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=hparams.lr, momentum=hparams.momentum, weight_decay=hparams.weight_decay)

        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(hparams.epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                if drop_method == 'Variational': 
                    outputs, kld_loss = model(inputs)
                    loss_r = criterion(outputs, labels)
                    loss = loss_r + kld_loss
                elif drop_method == 'Concrete':
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Add regularization
                    total_regularization = 0
                    for layer in model.modules():
                        if isinstance(layer, ConcreteDropout):
                            total_regularization += layer.regularisation
                    loss += total_regularization
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)

            # Evaluation phase
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    if drop_method == 'Variational':
                        outputs, kld_loss = model(inputs)
                        loss_r = criterion(outputs, labels)
                        test_loss = loss_r + kld_loss
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                test_loss /= len(test_loader)
                test_accuracy = 100 * correct / total
                test_accuracies.append(test_accuracy)
                test_losses.append(test_loss)

            # Print epoch results
            print(f"Epoch [{epoch+1}/{hparams.epochs}]  |  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%  |  Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.2f}%")
            print('----------------------------------------------------------------------------------------------------------')

            # Save the results for this dropout value
            results[dropout_rate] = {
                'test_loss': test_losses,
                'validation_accuracy': test_accuracies
            }

            # Save the model at the last epoch
            if epoch == hparams.epochs - 1:
                print('==> Saving model ...')
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoints'):
                    os.makedirs('checkpoints')  # Create checkpoint directory
                torch.save(state, f'./checkpoints/Dropout_method[{drop_method}]_dropout_rate_[{dropout_rate}]_ckpt.pth')

                save_results('results', drop_method, dropout_rate, test_accuracies)

                curr_acc = test_accuracies[-1]
                if best_acc < curr_acc:
                    best_droprate = dropout_rate
                    best_test_accuracies = test_accuracies
                    best_acc = curr_acc

    plot_results(results, drop_method)
    save_results('best_results', drop_method, best_droprate, best_test_accuracies)

#    return best_test_accuracies, best_droprate


    


def save_results(file, method_name, dropout_rate, accuracies_list):
    curr_results = {
        method_name: { 
            'dropout_rate': dropout_rate,
            'validation_accuracy': accuracies_list
        }
    }
    if not os.path.isdir(file):
        os.mkdir(file)

    if file == 'results':
        results_path = f'./results/{method_name}/Dropout_method[{method_name}]_dropout_rate_[{dropout_rate}].json'
        if os.path.exists(results_path):
            with open(results_path, 'w') as file:
                json.dump(curr_results, file, indent=4)


    elif file == 'best_results':
        all_best_results = {}
        # Check if the best results file exists
        best_results_path = f'./results/all_best_results.json'
        if os.path.exists(best_results_path):
            with open(best_results_path, 'r') as best_results_file:
                all_best_results = json.load(best_results_file)

        # Update the best results with the current method's best results
        all_best_results.update(curr_results)

        # Save the updated best results
        with open(best_results_path, 'w') as best_results_file:
            json.dump(all_best_results, best_results_file, indent=4)
    



def plot_results(results, drop_method):
    epochs = range(len(next(iter(results.values()))['validation_accuracy']))
    
    plt.figure(figsize=(10, 5))

    # Subplot for Accuracy
    plt.subplot(1, 2, 1)
    for dropout_rate, metrics in results.items():
        if (drop_method=='Variational'):
            plt.plot(epochs, metrics['validation_accuracy'], label=f'(log_sigma^2, threshold)= {dropout_rate}')
        elif (drop_method=='Concrete'):
            plt.plot(epochs, metrics['validation_accuracy'], label=f'l: {dropout_rate}')
        elif (drop_method=='ShakeDrop'):
            plt.plot(epochs, metrics['validation_accuracy'], label=f'alpha: {dropout_rate[0]}, beta: {dropout_rate[1]}')
        else:
            plt.plot(epochs, metrics['validation_accuracy'], label=f'Dropout: {dropout_rate}')
    plt.suptitle( f'Results (Dropout: {drop_method})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation - Accuracy per Epoch')
    plt.legend()
    plt.grid()

    # Subplot for Loss
    plt.subplot(1, 2, 2)
    for dropout_rate, metrics in results.items():
        if (drop_method=='Variational'):
            plt.plot((torch.Tensor(epochs)).cpu().numpy(), (torch.Tensor(metrics['test_loss'])).cpu().numpy(), label=f'Dropout: {dropout_rate}')
        else: 
            plt.plot((torch.Tensor(epochs)).cpu().numpy(),(torch.Tensor(metrics['test_loss'])).cpu().numpy(), label=f'Dropout: {dropout_rate}')
            

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation - Loss per Epoch')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()



def plot_best_results(results):
    epochs = range(len(next(iter(results.values()))['validation_accuracy']))

    plt.figure(figsize=(12, 6))

    # Subplot for Accuracy
    plt.subplot(1, 2, 1)
    for dropout_rate, metrics in results.items():
        plt.plot(epochs, metrics['validation_accuracy'], label=f'Val (Dropout: {dropout_rate})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid()




def choose_best_method(best_results):
    best_method = None
    best_final_accuracy = 0.0

    for method, results in best_results.items():
        final_accuracy = results['validation_accuracy'][-1]
        if final_accuracy > best_final_accuracy:
            best_final_accuracy = final_accuracy
            best_method = method
            best_drop_rate = results['dropout_rate']

    return best_method, best_drop_rate 



def load_model(model_class, best_dropout_rate, drop_method, N):
    #choose the best method and dropout rate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_class(dropout_rate=best_dropout_rate).to(device)

    if drop_method =='Concrete':
        l = best_dropout_rate
        wr = l**2. / N
        dr = 2. / N
        model = model_class(dropout_rate = 0.0, num_classes=7, weight_regulariser=wr,dropout_regulariser=dr).to(device)

    # load model, calculate accuracy and confusion matrix
    state = torch.load(f'./checkpoints/Dropout_method[{drop_method}]_dropout_rate_[{best_dropout_rate}]_ckpt.pth', map_location=device)
    model.load_state_dict(state['net'])
    return model