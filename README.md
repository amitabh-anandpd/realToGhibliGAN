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
    <td><img width="300" src="/images/original_image5.png"/></td>
    <td><img width="300" src="/images/predicted_image45.png"/></td>
  </tr>
</table>
