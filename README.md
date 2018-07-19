# Stylebank

Implementation of the paper: [StyleBank: An Explicit Representation for Neural Image Style Transfer](https://arxiv.org/abs/1703.09210)

# Implemented
* Training
* Incremental Learning

# Training
1. Prepare your content image dataset (I use MS-COCO)
2. Prepare your style image dataset

It takes about 1 days to train on a GeForce GTX 1080 Ti
# Examples

Style | Source | Transfered
:--:|:--:|:--:
<img src="style/00.jpg" width="256"> | <img src="examples/mountain.jpg" width="512"> | <img src="examples/00.jpg" width="512">
<img src="style/01.jpg" width="256"> | <img src="examples/japan.jpg" width="512"> | <img src="examples/01_japan.jpg" width="512">
<img src="style/02.jpg" width="256"> | <img src="examples/bridge.jpg" width="512"> | <img src="examples/02_bridge.jpg" width="512">
<img src="style/02.jpg" width="256"> | <img src="examples/japan101.jpg" width="512"> | <img src="examples/02_japan101.jpg" width="512">
<img src="style/09.jpg" width="256"> | <img src="examples/deer.jpg" width="512"> | <img src="examples/09_deer.jpg" width="512">
