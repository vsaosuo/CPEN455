import time
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import wandb
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths
from dataset import my_bidict

NUM_EPOCHS_SAMPLE_ACCURACY = 5
# changelog: add the get accuracy function
# Note: this function is derived from the get_label function in classification_evaluation.py
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
    nag_log_likelihood_of_classes = torch.stack(nag_log_likelihood_of_classes, dim=1).to(device) #[batch_size, num_classes]
    output = torch.argmin(nag_log_likelihood_of_classes, dim=1) # [batch_size] 

    return output

def train_or_test(model, data_loader, optimizer, loss_op, device, args, epoch, mode = 'training'):
    if mode == 'training':
        model.train()
    else:
        model.eval()
        
    deno =  args.batch_size * np.prod(args.obs) * np.log(2.)        
    loss_tracker = mean_tracker()
    acc_tracker = ratio_tracker()

    for batch_idx, item in enumerate(tqdm(data_loader)):
        # changelog: add the label of the image to the model
        model_input, labels = item
        model_input = model_input.to(device)
        model_output = model(model_input, labels)
        loss = loss_op(model_input, model_output)
        loss_tracker.update(loss.item()/deno)
        if mode == 'training':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # elif mode == 'val' and epoch % NUM_EPOCHS_SAMPLE_ACCURACY == 0:
        #     # changelog: add the label of the image for validation
        #     original_label = [my_bidict.get(label, 0) for label in labels]
        #     original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        #     answer = get_label(model, model_input, device)
        #     correct_num = torch.sum(answer == original_label)
        #     acc_tracker.update(correct_num.item(), model_input.shape[0])
        #     print("original_label: ", original_label)
        #     print("answer: ", answer)
        
    if args.en_wandb:
        wandb.log({mode + "-Average-BPD" : loss_tracker.get_mean()})
        wandb.log({mode + "-epoch": epoch})
        # if mode == 'val' and epoch % NUM_EPOCHS_SAMPLE_ACCURACY == 0:
        #     wandb.log({mode + "-Accuracy" : acc_tracker.get_ratio()})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-w', '--en_wandb', type=bool, default=False,
                            help='Enable wandb logging')
    parser.add_argument('-t', '--tag', type=str, default='default',
                            help='Tag for this run')
    
    # sampling
    parser.add_argument('-c', '--sampling_interval', type=int, default=5,
                        help='sampling interval')
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-sd', '--sample_dir',  type=str, default='samples',
                        help='Location for saving samples')
    parser.add_argument('-d', '--dataset', type=str,
                        default='cpen455', help='Can be either cifar|mnist|cpen455')
    parser.add_argument('-st', '--save_interval', type=int, default=10,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('--obs', type=tuple, default=(3, 32, 32),
                        help='Observation shape')
    
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=1,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=40,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=5,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.0002, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')
    parser.add_argument('-sb', '--sample_batch_size', type=int, default=32,
                        help='Batch size during sampling per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=5000, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    check_dir_and_create(args.save_dir)
    
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model_name = 'pcnn_' + args.dataset + "_"
    model_path = args.save_dir + '/'
    if args.load_params is not None:
        model_name = model_name + 'load_model'
        model_path = model_path + model_name + '/'
    else:
        model_name = model_name + 'from_scratch'
        model_path = model_path + model_name + '/'
    
    job_name = "PCNN_Training_" + "dataset:" + args.dataset + "_" + args.tag
    
    if args.en_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set entity to specify your username or team name
            # entity="qihangz-work",
            # set the wandb project where this run will be logged
            project="CPEN455HW",
            # group=Group Name
            name=job_name,
        )
        wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        wandb.config.update(args)

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    #Reminder: if you have patience to read code line by line, you should notice this comment. here is the reason why we set num_workers to 0:
    #In order to avoid pickling errors with the dataset on different machines, we set num_workers to 0.
    #If you are using ubuntu/linux/colab, and find that loading data is too slow, you can set num_workers to 1 or even bigger.
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':True}

    # set data
    if "mnist" in args.dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), rescaling, replicate_color_channel])
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                            train=True, transform=ds_transforms), batch_size=args.batch_size, 
                                shuffle=True, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    elif "cifar" in args.dataset:
        ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
        if args.dataset == "cifar10":
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
                download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
            
            test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
        elif args.dataset == "cifar100":
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=True, 
                download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
            
            test_loader  = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            raise Exception('{} dataset not in {cifar10, cifar100}'.format(args.dataset))
    
    elif "cpen455" in args.dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
        train_loader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                  mode = 'train', 
                                                                  transform=ds_transforms), 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   **kwargs)
        test_loader  = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                  mode = 'test', 
                                                                  transform=ds_transforms), 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   **kwargs)
        val_loader  = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                  mode = 'validation', 
                                                                  transform=ds_transforms), 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   **kwargs)
    else:
        raise Exception('{} dataset not in {mnist, cifar, cpen455}'.format(args.dataset))
    
    args.obs = (3, 32, 32)
    input_channels = args.obs[0]
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
    model = model.to(device)

    if args.load_params:
        model.load_state_dict(torch.load(args.load_params))
        print('model parameters loaded')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    
    for epoch in tqdm(range(args.max_epochs)):
        train_or_test(model = model, 
                      data_loader = train_loader, 
                      optimizer = optimizer, 
                      loss_op = loss_op, 
                      device = device, 
                      args = args, 
                      epoch = epoch, 
                      mode = 'training')
        
        # decrease learning rate
        scheduler.step()
        train_or_test(model = model,
                      data_loader = test_loader,
                      optimizer = optimizer,
                      loss_op = loss_op,
                      device = device,
                      args = args,
                      epoch = epoch,
                      mode = 'test')
        
        train_or_test(model = model,
                      data_loader = val_loader,
                      optimizer = optimizer,
                      loss_op = loss_op,
                      device = device,
                      args = args,
                      epoch = epoch,
                      mode = 'val')
        
        if epoch % args.sampling_interval == 0:
            for key in my_bidict.keys():
                
                # changelog: add the label of the image for sampling
                print(f'......sampling {key}......')
                labels = [key] * args.sample_batch_size

                sample_t = sample(model, args.sample_batch_size, args.obs, sample_op, labels)
                sample_t = rescaling_inv(sample_t)
                save_images(sample_t, args.sample_dir, f"{key}_epoch{epoch}")
                sample_result = wandb.Image(sample_t, caption="key - {} - epoch {}".format(key, epoch))
                
                gen_data_dir = args.sample_dir
                ref_data_dir = args.data_dir +'/test'
                paths = [gen_data_dir, ref_data_dir]
                try:
                    fid_score = calculate_fid_given_paths(paths, 32, device, dims=192)
                    print("Dimension {:d} works! fid score: {}".format(192, fid_score))
                except:
                    print("Dimension {:d} fails!".format(192))
                    
                if args.en_wandb:
                    wandb.log({"samples": sample_result,
                                "FID": fid_score})
        
        if (epoch + 1) % args.save_interval == 0: 
            if not os.path.exists("models"):
                os.makedirs("models")
            torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))

## Training Model
# | Date       | Time     | Epoch | Train BPD | Test BPD  | Val BPD   | Run ID                        | Evalations (validation, FID) |
# |------------|----------|-------|-----------|-----------|-----------|-------------------------------|------------------------------|
# | 2024-04-09 | 5:16:10  | 500   | 3.17      | 3.03      | 3.04      | Fully-Trained-Middle-Fusion   | (0.6551, 24.2779)            |
