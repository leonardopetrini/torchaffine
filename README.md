# TorchAffine
Apply affine transformations (translation, rotation, compression, pure shear) on images in PyTorch.

Given `images ~ [batch_size, ch, n, n]`,
```
transformed_images = transform(images, translation_x, translation_y, rotation, compression,
              shear_a, shear_b, center_x, center_y)
```

Transformations are computed by applying a displacement field `tau` to pixels.

Translations give a constant displacement of components `T = [tx, ty]`.
 
Linear transformations are related to the gradients of `tau` that in 2D are represented by a `2x2` matrix:
```
U = [[\partial_x \tau_x, \partial_y \tau_x]
     [\partial_x \tau_y, \partial_y \tau_y]].
```
Given a pixel position `X = [x, y]`, its transformed position is then given by
```
X' = (U + I) X + T
```
where `I` is the identity, `T` the translation vector and `U` the linear transformations matrix that can take the following forms: 
- Rotation:
```
R = [[cos t - 1, -sin t]
     [sin t,  cos t - 1]]
```
- Compression:
```
C = [[s, 0]
     [0, s]]
```
- Pure shear:
```
S = [[a,  b]
     [b, -a]]
```

### Install
```
python -m pip install git+https://github.com/leonardopetrini/torchaffine.git
```
### Usage
Examples of how to use the code are reported `examples.ipynb`.
### Dependencies
- `torch`

For the example:
- `matplotlib`

cf. `requirements.txt` for versions.