# Generative Adversarial Network Art 🎨
Creating a Generative Adversarial Network to generate modern art and training it    |    Numpy, PIL, keras

## Concept 🔎

Generative adversarial networks (GANs) are algorithmic architectures that use two neural networks, pitting one against the other in order to generate new, synthetic instances of data that can pass for real data.

## Steps 🐌

1 - Get from the catalog of the Web Galery of Art (https://www.wga.hu/) some dataset
    
   We set some features and get the URLs with the `downloader.py` script
    
   ```Python
   form_feature = "painting"
   type_feature = "landscape"
   ```
   
   This parameters above gave us almost 3000 pictures.

   ![alt text](https://github.com/ThomasCochou/Generative_Adversarial_Network_Art/blob/master/wga_exemple/1lands11.jpg?raw=true)
   <p align="center"><i>
   Le golfe de Marseille vu de l’Estaque, par Paul Cézanne
   </i></p>

2 - Resize the images from the dataset with `resizer.py` to 300x300

3 - Train the models with `art_gan.py` and see the results each 100 epochs in the `output\` folder

   ![alt text](https://github.com/ThomasCochou/Generative_Adversarial_Network_Art/blob/master/output/trained-7.png?raw=true)
    
   ![alt text](https://github.com/ThomasCochou/Generative_Adversarial_Network_Art/blob/master/output/trained-12.png?raw=true)
    
   ![alt text](https://github.com/ThomasCochou/Generative_Adversarial_Network_Art/blob/master/output/trained-23.png?raw=true)

## Spell 💻

It is possible to use Spell (https://spell.ml/) to compute the program online

```
spell login
Spell upload wga.npy
Spell run python art_gan.py -t cpu -m uploads/art_gan_3000/wga.npy
```

## Problems 🔧

Looks like the accuracies of both models tends to 100% very fast, and the results are mysterious.

```
Mean discriminator accuracy: 99.3514311650107, Mean generator accuracy: 99.41208541846206
10000 epoch, Discriminator accuracy: 100.0, Generator accuracy: 100.0
```

## Ressources 📚
https://towardsdatascience.com/generating-modern-arts-using-generative-adversarial-network-gan-on-spell-39f67f83c7b4
https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/
https://towardsdatascience.com/generating-abstract-art-using-gans-with-keras-153b7f11bd0
https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
