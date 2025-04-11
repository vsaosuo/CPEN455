'''
This code is used to evaluate the FID score of the generated images.
You should at least guarantee this code can run without any error on test set.
And whether this code can run is the most important factor for grading.
We provide the remaining code,  you can't modify other code, all you should do are:
1. Modify the sample function to get the generated images from the model and ensure the generated images are saved to the gen_data_dir(line 20-33)
2. Modify how you call your sample function(line 50-55)

REQUIREMENTS:
- You should save the generated images to the gen_data_dir, which is fixed as './samples'
- If you directly run this code, it should generate images and calculate the FID score, you should follow the same format as the demonstration, there should be 100 images in 4 classes, each class has 25 images
'''
from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
from model import * 
from dataset import *
import os
import torch
import argparse
#TODO: Begin of your code
import csv
# This is a demonstration of how to call the sample function, feel free to modify it
# You should modify this sample function to get the generated images from your model
# You should save the generated images to the gen_data_dir, which is fixed as 'samples'
sample_op = lambda x : sample_from_discretized_mix_logistic(x, 10) # 10 is the number of mixtures which follows nr_logistic_mix
def my_sample(model, gen_data_dir, sample_batch_size = 25, obs = (3,32,32), sample_op = sample_op):
    for key in my_bidict.keys():
        print(f"Label: {key}")
        labels = [key] * sample_batch_size

        #generate images for each label, each label has 25 images
        sample_t = sample(model, sample_batch_size, obs, sample_op, labels)
        sample_t = rescaling_inv(sample_t)
        save_images(sample_t, os.path.join(gen_data_dir), label=key)
    pass
# End of your code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ref_data_dir', type=str,
                        default="data/test", help='Location for the dataset')
    parser.add_argument('-md', '--model', type=str,
                        default='models/conditional_pixelcnn.pth', help='Model to load')
    parser.add_argument('-sp', '--samples_path', type=str,
                        default='samples', help='Model to load')
    
    args = parser.parse_args()
    
    ref_data_dir = args.ref_data_dir
    # TODO: Changed from original
    gen_data_dir = os.path.join(os.path.dirname(__file__), args.samples_path)
    BATCH_SIZE=128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)

    #TODO: Begin of your code
    #Load your model and generate images in the gen_data_dir, feel free to modify the model
    model = PixelCNN(nr_resnet=5, nr_filters=160, nr_logistic_mix=10, input_channels=3)
    model = model.to(device)

    # Load pretrained model
    # Note: this code is copied from classification_evaluation.py
    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if os.path.exists(model_path):
        # TODO: Changed from original to fit with MAC
        # model.load_state_dict(torch.load(model_path))
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = model.eval()
    #End of your code
    
    my_sample(model=model, gen_data_dir=gen_data_dir)
    
    paths = [gen_data_dir, ref_data_dir]
    print("#generated images: {:d}, #reference images: {:d}".format(
        len(os.listdir(gen_data_dir)), len(os.listdir(ref_data_dir))))

    try:
        fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score, gen_data_dir))
    except:
        print("Dimension {:d} fails!".format(192))
        
    print("Average fid score: {}".format(fid_score))

    # TODO: Save the accuracy to a csv file
    with open('fid.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Only write headers if file is empty
        if file.tell() == 0:
            writer.writerow(['Model', 'fib_score'])
        writer.writerow([args.model, fid_score])
