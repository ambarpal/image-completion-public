# Image Completion
This repository contains the codebase for the Image Completion project done for the [Computer Vision 2017 Course](https://sites.google.com/a/iiitd.ac.in/cv) at IIIT Delhi. 

## Authors
The codebase is maintained by [Ambar Pal](https://github.com/ambarpal) and [Aishwarya Jaiswal](https://github.com/aishblue)

## How to Run
We will first train a GAN on the original data distribution and then use the trained model to perform image inpainting on corrupted images

### Train GAN to learn the data distribution
``python train.py --dataset .. --num_train_epochs .. --num_disc_steps .. --num_gen_steps .. --batch_size .. --save_checkpoint_every .. --generate_samples_every .. --flip_alpha .. ``

  The following configurations work best for the various datasets:
  * ``python train.py --dataset 'MNIST' --num_train_epochs 10 --num_disc_steps 1 --num_gen_steps 2 --batch_size 64 --save_checkpoint_every 250 --generate_samples_every 100 --flip_alpha 0.3 ``

  * ``python --dataset 'CIFAR10' --num_train_epochs 50 --num_disc_steps 1 --num_gen_steps 1 --batch_size 64 --save_checkpoint_every 250 --generate_samples_every 100 --flip_alpha 0.2``
    
  * ``python --dataset 'CELEBA' --num_train_epochs 15 --num_disc_steps 1 --num_gen_steps 1 --batch_size 64 --save_checkpoint_every 250 --generate_samples_every 100 --flip_alpha 0.2``
        
  * ``python --dataset 'SVHN' --num_train_epochs 15 --num_disc_steps 1 --num_gen_steps 1 --batch_size 64 --save_checkpoint_every 250 --generate_samples_every 100 --flip_alpha 0.3``
