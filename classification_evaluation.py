'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
def get_label(model, model_input, device):
    # Write your code here, replace the random classifier with your trained model
    # and return the predicted label, which is a tensor of shape (batch_size,)
    batch_size = model_input.shape[0]

    nag_log_likelihood_of_classes = []

    # For P(class|image) = P(image|class) × P(class),
    # Since we assume a uniform prior (which is the P(class) term), 
    # we only need to worry about maximizing the P(image|class) term.
    for key in my_bidict.keys():
        labels = [key] * batch_size
        
        # Get the predicted image from the model
        generated_image = model(x=model_input, labels=labels)

        # Compute the log-likelihood of the generated image (i.e., P(image|class))
        # Note: this function is taken from pcnn_train.py
        loss_op = lambda real, fake : discretized_mix_logistic_loss_per_image(real, fake)
        neg_log_likelihood = loss_op(model_input, generated_image)

        # Store the negative log-likelihood for the current class
        nag_log_likelihood_of_classes.append(neg_log_likelihood)
    
    # Since we dealing with nagative log-likelihood, we need to take the minizing of P(image|class)
    # to get the maximum likelihood
    nag_log_likelihood_of_classes = torch.stack(nag_log_likelihood_of_classes, dim=1) #[batch_size, num_classes]
    output = torch.argmin(nag_log_likelihood_of_classes, dim=1) # [batch_size] 

    return output

# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    parser.add_argument('-md', '--model', type=str,
                        default='models/conditional_pixelcnn.pth', help='Model to load')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #TODO:Begin of your code
    #You should replace the random classifier with your trained model
    # model = random_classifier(NUM_CLASSES)
    model = PixelCNN(nr_resnet=5, nr_filters=160, nr_logistic_mix=10, input_channels=3)
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    # TODO: change to original
    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if os.path.exists(model_path):
        # TODO: Changed from original to fit with MAC
        model.load_state_dict(torch.load(model_path))
        # model.load_state_dict(torch.load(model_path, map_location=device))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")
    
    # TODO: Save the accuracy to a csv file
    with open('accuracy.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Only write headers if file is empty
        if file.tell() == 0:
            writer.writerow(['Model', 'Accuracy'])
        writer.writerow([args.model, acc])
        