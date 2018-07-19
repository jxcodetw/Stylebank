from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Sequential
import torchvision.models as models
import args

device = args.device
vgg16 = models.vgg16(pretrained=True).features.to(device).eval()

class ContentLoss(nn.Module):

	def __init__(self):
		super(ContentLoss, self).__init__()
		# we 'detach' the target content from the tree used
		# to dynamically compute the gradient: this is a stated value,
		# not a variable. Otherwise the forward method of the criterion
		# will throw an error.
		# self.target = target
		self.target = None
		self.mode = 'learn'

	def forward(self, input):
		if self.mode == 'loss':
			self.loss = self.weight * F.mse_loss(input, self.target)
		elif self.mode == 'learn':
			self.target = input.detach()
		return input

def gram_matrix(input):
	a, b, c, d = input.size()  # a=batch size(=1)
	# b=number of feature maps
	# (c,d)=dimensions of a f. map (N=c*d)

	features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

	G = torch.mm(features, features.t())  # compute the gram product

	# we 'normalize' the values of the gram matrix
	# by dividing by the number of element in each feature maps.
	return G.div(a * b * c * d)

class StyleLoss(nn.Module):

	def __init__(self):
		super(StyleLoss, self).__init__()
		self.targets = []
		# self.target = gram_matrix(target_feature).detach()
		self.mode = 'learn'

	def forward(self, input):
		G = gram_matrix(input)
		if self.mode == 'loss':
			self.loss = self.weight * F.mse_loss(G, self.target)
		elif self.mode == 'learn':
			self.target = G.detach()
		return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
	def __init__(self):
		super(Normalization, self).__init__()
		# .view the mean and std to make them [C x 1 x 1] so that they can
		# directly work with image Tensor of shape [B x C x H x W].
		# B is batch size. C is number of channels. H is height and W is width.
		mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
		std = torch.tensor([0.229, 0.224, 0.225]).to(device)
		self.mean = torch.tensor(mean).view(-1, 1, 1)
		self.std = torch.tensor(std).view(-1, 1, 1)

	def forward(self, img):
		# normalize img
		return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_layers = ['conv_9']
content_weight = {
	'conv_9': 1
}
style_layers = [ 'conv_2', 'conv_4', 'conv_6', 'conv_9']
style_weight = {
	'conv_2': 1,
	'conv_4': 1,
	'conv_6': 1,
	'conv_9': 1,
}

class LossNetwork(nn.Module):

	def __init__(self):
		super(LossNetwork, self).__init__()
		cnn = deepcopy(vgg16)
		normalization = Normalization().to(device)
		# just in order to have an iterable access to or list of content/syle
		# losses
		content_losses = []
		style_losses = []

		# assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
		# to put in modules that are supposed to be activated sequentially
		model = nn.Sequential(normalization)

		i = 0  # increment every time we see a conv
		for layer in cnn.children():
			if isinstance(layer, nn.Conv2d):
				i += 1
				name = 'conv_{}'.format(i)
			elif isinstance(layer, nn.ReLU):
				name = 'relu_{}'.format(i)
				# The in-place version doesn't play very nicely with the ContentLoss
				# and StyleLoss we insert below. So we replace with out-of-place
				# ones here.
				layer = nn.ReLU(inplace=False)
			elif isinstance(layer, nn.MaxPool2d):
				name = 'pool_{}'.format(i)
			elif isinstance(layer, nn.BatchNorm2d):
				name = 'bn_{}'.format(i)
			else:
				raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

			model.add_module(name, layer)

			if name in content_layers:
				# add content loss:
				# target_feature = model(content_img).detach()
				content_loss = ContentLoss()
				content_loss.weight = content_weight[name]
				model.add_module("content_loss_{}".format(i), content_loss)
				content_losses.append(content_loss)

			if name in style_layers:
				# add style loss:
				# target_feature = model(style_img).detach()
				style_loss = StyleLoss()
				style_loss.weight = style_weight[name]
				model.add_module("style_loss_{}".format(i), style_loss)
				style_losses.append(style_loss)

		
		
		# now we trim off the layers after the last content and style losses
		for i in range(len(model) - 1, -1, -1):
			if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
				break

		model = model[:(i + 1)]

		self.model = model
		self.style_losses = style_losses
		self.content_losses = content_losses

	def learn_content(self, input):
		for cl in self.content_losses:
			cl.mode = 'learn'
		for sl in self.style_losses:
			sl.mode = 'nop'
		self.model(input) # feed image to vgg19
	
	def learn_style(self, input):
		for cl in self.content_losses:
			cl.mode = 'nop'
		for sl in self.style_losses: 
			sl.mode = 'learn'
		self.model(input) # feed image to vgg19

	def forward(self, input, content, style):
		self.learn_content(content)
		self.learn_style(style)

		for cl in self.content_losses:
			cl.mode = 'loss'
		for sl in self.style_losses:
			sl.mode = 'loss'
		self.model(input) # feed image to vgg19

		content_loss = 0
		style_loss = 0

		for cl in self.content_losses:
			content_loss += cl.loss
		for sl in self.style_losses:
			style_loss += sl.loss

		return content_loss, style_loss

class StyleBankNet(nn.Module):
	def __init__(self, total_style):
		super(StyleBankNet, self).__init__()
		self.total_style = total_style
		
		self.encoder_net = Sequential(
			nn.Conv2d(3, 32, kernel_size=(9, 9), stride=2, padding=(4, 4), bias=False),
			nn.InstanceNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=(1, 1), bias=False),
			nn.InstanceNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
			nn.InstanceNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
			nn.InstanceNorm2d(256),
			nn.ReLU(inplace=True),
		)
		
		self.decoder_net = Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
			nn.InstanceNorm2d(128),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
			nn.InstanceNorm2d(64),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=(1, 1), bias=False),
			nn.InstanceNorm2d(32),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(32, 3, kernel_size=(9, 9), stride=2, padding=(4, 4), bias=False),
		)
		
		self.style_bank = nn.ModuleList([
			Sequential(
				nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
				nn.InstanceNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
				nn.InstanceNorm2d(256),
				nn.ReLU(inplace=True)
			)
			for i in range(total_style)])
		
	def forward(self, X, style_id=None):
		z = self.encoder_net(X)
		if style_id is not None:
			new_z = []
			for idx, i in enumerate(style_id):
				zs = self.style_bank[i](z[idx].view(1, *z[idx].shape))
				new_z.append(zs)
			z = torch.cat(new_z, dim=0)
			# z = self.bank_net(z)
		return self.decoder_net(z)
