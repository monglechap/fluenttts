import torch, pdb
import torch.nn as nn
import torch.nn.functional as F

class AffineLinear(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(AffineLinear, self).__init__()
		affine = nn.Linear(in_dim, out_dim)
		self.affine = affine

	def forward(self, input_data):
		return self.affine(input_data)

class StyleAdaptiveLayerNorm(nn.Module):
	def __init__(self, in_channel, style_dim):
		super(StyleAdaptiveLayerNorm, self).__init__()
		self.in_channel = in_channel
		self.norm = nn.LayerNorm(in_channel, elementwise_affine=False)

		self.style = AffineLinear(style_dim, in_channel * 2)
		self.style.affine.bias.data[:in_channel] = 1
		self.style.affine.bias.data[in_channel:] = 0

	def forward(self, input_data, style_code):
		style = self.style(style_code)
	
		gamma, beta = style.chunk(2, dim=-1)

		out = self.norm(input_data)
		out = gamma * out + beta

		return out

