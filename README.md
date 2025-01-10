# EfficientNet
This repo provides a clean implementation of EfficientNet.

## TODO
- fourier feature mapped positional encoding (maintaining unscaled aspect ratio)
- decide whether a pixel can be marked as multiple masks or not
    - if it can be, the comparison code needs to be `&` not `==`
    - if not, just label each pixel with the appropiate index
    - and either way, the maximum number of masks should be encoded differently
- samples by pixels without a mask shouldn't add to loss
- train ResUNet for pixelwise embedding and use UMAP to color the output
- Stop [this](https://stackoverflow.com/a/58385932) from happening (?)
- Test TPU compat
