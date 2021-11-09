# Generative Adversarial Network Art 🎨
Creating a Generative Adversarial Network to generate modern art and training it    |    Numpy, PIL, keras

# Concept 🔎

Generative adversarial networks (GANs) are algorithmic architectures that use two neural networks, pitting one against the other in order to generate new, synthetic instances of data that can pass for real data.

# Steps 🐌

1 - Get from the catalog of the Web Galery of Art (https://www.wga.hu/) some dataset
    We set some features and get the URLs with the `downloader.py` script
    ``Python
    form_feature = "painting"
    type_feature = "landscape"
    timeframe_feature = "1851-1900"
    ``
    This parameters gave us 462 pictures.
    
    EXEMPLE HERE

2 - Resize the images from the dataset with `resizer.py` to 300x300

3 - Train the models with `art_gan.py` and see the results each 100 epochs in the `output\` folder

    EXEMPLE HERE


# Problems 😢

Looks like the accuracies of both models tends to 100% very fast, and the results are pretty weird. But in a way, it's ART.

# Ressources 📚
https://towardsdatascience.com/generating-modern-arts-using-generative-adversarial-network-gan-on-spell-39f67f83c7b4
https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/
https://towardsdatascience.com/generating-abstract-art-using-gans-with-keras-153b7f11bd0
https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
