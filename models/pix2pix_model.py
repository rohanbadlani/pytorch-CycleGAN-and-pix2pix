import torch
from .base_model import BaseModel
from . import networks

#import pretrained vgg net
import torchvision.models as models
from .CX_distance import symetric_CX_loss, CX_loss
import pdb
import torch.nn as nn
from .perceptual_loss_utils import normalize_batch, perceptual_loss
from torchsummary import summary
from collections import namedtuple

class Vgg16Features(torch.nn.Module):
    #use for perceptual
    def __init__(self, requires_grad=False):
        super(Vgg16Features, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).cuda().features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        #pdb.set_trace()

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class Vgg19Features(torch.nn.Module):
    #use for contextual
    def __init__(self, requires_grad=False):
        super(Vgg19Features, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).cuda().features
        self.conv1_2 = torch.nn.Sequential()
        self.conv2_2 = torch.nn.Sequential()
        self.conv3_2 = torch.nn.Sequential()
        self.conv3_4 = torch.nn.Sequential()
        self.conv4_2 = torch.nn.Sequential()
        self.conv4_4 = torch.nn.Sequential()
        self.conv5_2 = torch.nn.Sequential()
        self.conv5_4 = torch.nn.Sequential()
        
        for x in range(4):
            self.conv1_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.conv2_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):
            self.conv3_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 18):
            self.conv3_4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 23):
            self.conv4_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 27):
            self.conv4_4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 32):
            self.conv5_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(32, 36):
            self.conv5_4.add_module(str(x), vgg_pretrained_features[x])

        #pdb.set_trace()

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.conv1_2(X)
        h_conv1_2 = h
        h = self.conv2_2(h)
        h_conv2_2 = h
        h = self.conv3_2(h)
        h_conv3_2 = h
        h = self.conv3_4(h)
        h_conv3_4 = h
        h = self.conv4_2(h)
        h_conv4_2 = h
        h = self.conv4_4(h)
        h_conv4_4 = h
        h = self.conv5_2(h)
        h_conv5_2 = h
        h = self.conv5_4(h)
        h_conv5_4 = h
        
        vgg_outputs = namedtuple("VggOutputs", ['conv1_2', 'conv2_2', 'conv3_2', 'conv3_4', 'conv4_2', 'conv4_4', 'conv5_2', 'conv5_4'])
        out = vgg_outputs(h_conv1_2, h_conv2_2, h_conv3_2, h_conv3_4, h_conv4_2, h_conv4_4, h_conv5_2, h_conv5_4)
        return out

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if not opt.not_conditional:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                print("Creating unconditional discriminator")
                self.netD = networks.define_D( opt.output_nc, opt.ndf, opt.netD,
                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
        # For perceptual Loss.
        self.vgg16_features = Vgg16Features(requires_grad=False)
        self.perceptual_loss_layers = ['relu3_3']

        # For contextual Loss.
        self.vgg19_features = Vgg19Features(requires_grad=False)
        self.contextual_loss_layers = ['conv3_2', 'conv4_2']
        self.contextual_loss_layers_source = ['conv4_2']

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if self.opt.not_conditional:
            fake_AB = self.fake_B
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        if self.opt.not_conditional:
            real_AB = self.real_B
        else:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.not_conditional:
            fake_AB = self.fake_B
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        self.loss_G = self.loss_G_GAN
        
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        if "L1" in self.opt.loss_type:
            self.loss_G += self.loss_G_L1
        
        if "contextual" in self.opt.loss_type:
            real_vgg_features = self.vgg19_features(self.real_B)
            fake_vgg_features = self.vgg19_features(self.fake_B)

            self.loss_G_Contextual = 0.0

            for layer in self.contextual_loss_layers:
                self.loss_G_Contextual += symetric_CX_loss(getattr(real_vgg_features,layer), getattr(fake_vgg_features,layer))

            self.loss_G += self.loss_G_Contextual.squeeze() * self.opt.lambda_L1

            if "unpaired" in self.opt.loss_type:
                inputimg_vgg_features = self.vgg19_features(self.real_A)

                self.loss_G_Contextual_source = 0.0

                for layer in self.contextual_loss_layers_source:
                    self.loss_G_Contextual_source += symetric_CX_loss(getattr(inputimg_vgg_features,layer), getattr(fake_vgg_features,layer))

                self.loss_G += self.loss_G_Contextual_source.squeeze() * self.opt.lambda_L1 * len(self.contextual_loss_layers)/len(self.contextual_loss_layers_source)

        #Fourth is the Perceptual loss as the L1 distance between real vs fake features
        # 1. Normalize input images using ImageNet features
        if "perceptual" in self.opt.loss_type:
            real_normalized_image = normalize_batch(self.real_B)
            fake_normalized_image = normalize_batch(self.fake_B)

            # 2. Forward pass through VGG
            pl_real_vgg_features = self.vgg16_features(real_normalized_image)
            pl_fake_vgg_features = self.vgg16_features(fake_normalized_image)

            # 3. Use L1/L2 distance between VGG features. Currently L1 norm. Implementation inspired from https://github.com/tengteng95/Pose-Transfer/blob/master/losses/L1_plus_perceptualLoss.py
            self.perceptual_loss = 0.0

            for layer in self.perceptual_loss_layers:
                self.perceptual_loss += perceptual_loss(getattr(pl_real_vgg_features,layer), getattr(pl_fake_vgg_features,layer), 1)

            self.loss_G += self.perceptual_loss.squeeze() * self.opt.lambda_L1
            
            if "unpaired" in self.opt.loss_type:
                input_normalized_image = normalize_batch(self.real_A)
                pl_input_vgg_features = self.vgg16_features(input_normalized_image)

                self.perceptual_loss_source = 0.0

                for layer in self.perceptual_loss_layers:
                    self.perceptual_loss_source += perceptual_loss(getattr(pl_input_vgg_features,layer), getattr(pl_fake_vgg_features,layer), 1)

                self.loss_G += self.perceptual_loss_source.squeeze() * self.opt.lambda_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
