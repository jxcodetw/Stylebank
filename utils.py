import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import args
from time import sleep

def showimg(img):
	"""
	Input a pytorch image tensor with size (channel, width, height) and display it.
	"""
	img = img.clamp(min=0, max=1)
	img = img.cpu().numpy().transpose(1, 2, 0)
	plt.imshow(img)
	plt.show()

class Resize(object):
	"""
	Resize with aspect ration preserved.
	"""
	def __init__(self, size):
		self.size = size

	def __call__(self, img):
		m = min(img.size)
		new_size = (int(img.size[0] / m * self.size), int(img.size[1] / m * self.size))
		return img.resize(new_size, resample=Image.BILINEAR)

def adjust_learning_rate(optimizer, step):
	"""
	Learning rate decay
	"""
	lr = max(args.lr * (0.8 ** (step)), 1e-6)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

def get_sid_batch(style_id_seg, batch_size):
	ret = style_id_seg
	while len(ret) < batch_size:
		ret += style_id_seg
	
	return ret[:batch_size]

content_img_transform = transforms.Compose([
	Resize(513),
	transforms.RandomCrop([513, 513]),
	transforms.ToTensor(),
])

style_img_transform = transforms.Compose([
	Resize(513),
	transforms.CenterCrop([513, 513]),
	transforms.ToTensor(),
])