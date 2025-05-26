# realToGhibliGAN
This project implements a Generative Adversarial Network (GAN) for translating real-world images into Studio Ghibli-style anime images. The model uses a U-Net based Generator, a PatchGAN Discriminator, and a combination of L1 loss, adversarial loss, and perceptual loss to produce high-quality anime-style outputs.

Dataset <a href="https://www.kaggle.com/datasets/labledata/ghibli-dataset">link</a> (kaggle)

## Current model output - 
<table>
  <tr>
    <td>Original</td>
    <td>Model Output</td>
  </tr>
  <tr>
    <td><img width="300" src="/images/original_image2.jpg"/></td>
    <td><img width="300" src="/images/predicted_image42.png"/></td>
  </tr>
  <tr>
    <td><img width="300" src="/images/original_image3.jpg"/></td>
    <td><img width="300" src="/images/predicted_image43.png"/></td>
  </tr>
  <tr>
    <td><img width="300" src="/images/original_image5.jpg"/></td>
    <td><img width="300" src="/images/predicted_image45.png"/></td>
  </tr>
</table>


## Notes

As you can see from the output, it is not completely giving desired outputs. One reason might be downsampling. If we keep the resolution high, we might be able to retain the low level features like eyes and nose. But that requires more computation power and kaggle and google colab's free version are not sufficient. Hence, it is not yet complete.
