# Procedure to run the code


## Dataset

We are using the Flicker30k datasets. You can download from this link <http://shannon.cs.illinois.edu/DenotationGraph/>. We assum that flicker30k data named **flickr30k-images** is located at the same directly as the one where **create_dataset.py** is located.  

Andrej Karpathy's training, validation, and test splits <http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip> was used. This zip file contain the captions. You can find splits and captions for the Flicker30k datasets. Please save this folder at the same directly as the one **create_dataset.py** is located and name this folder **caption_datasets**.

## Create dataset for analysis

See the **create_dataset.py**. This reads the data downloaded and saves the following files –

Just type the following to run this code.

```
python create_dataset.py
```

- An HDF5 file containing images for each split in an I, 3, 256, 256 tensor, where I is the number of images in the split. Pixel values are still in the range [0, 255], and are stored as unsigned 8-bit Ints.
- A JSON file for each split with a list of N_c * I encoded captions, where N_c is the number of captions sampled per image. These captions are in the same order as the images in the HDF5 file. Therefore, the ith caption will correspond to the i // N_cth image.
- A JSON file for each split with a list of N_c * I caption lengths. The ith value is the length of the ith caption, which corresponds to the i // N_cth image.
- A JSON file which contains the word_map, the word-to-index dictionary.

These files are all saved inside **caption_datasets** directly.


## Model 

There are mainly two types of models: "show, attend and tell model" and "show and tell model". "show, attend and tell model" is located in **show-attend-and-tell** folder and "show and tell model" is located in **show-attend-and-tell** folder. 

First look at **model.py** (**show_and_tell_model.py** for show and tell model).

We used a pretrained ResNet-101 and ResNet-152 already available in PyTorch's torchvision module for encoder.

We used pretrained **glove.840B** for embedding. You can delete it if you want. If you use **glove.840B**, upload it to the directly where **create_dataset.py** is located. 

## Train

Make sure to run **create_dataset.py** beforehand. 

See **train.py** (**show_and_tell_train.py** for show and tell model).

To train your model from scratch, simply run this file –

```
python train.py
```

To resume training at a checkpoint, point to the corresponding file with the checkpoint parameter at the beginning of the code.

## Get caption 

See **get_caption.py**. (**show_and_tell_get_caption.py** for show and tell model)

To caption an image from the command line, please point to the path to image, path to model check point, path to word map like below.

```
python get_caption.py --img='../imagename.jpg' --model='BEST_checkpoint_flickr30k_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='../caption_datasets/WORDMAP_flickr30k_5_cap_per_img_5_min_word_freq.json' --beam_size=20
```

## Get Blue1 and Blue4 score

See **eval.py** (**show_and_tell_eval.py** for show and tell model)

Give the path to the checkpoint at the beginning of the code and run the code like this

```
python eval.py
```

This will return the number of epoch of the checkpoint, BLUE4 and BLUE1 score.
