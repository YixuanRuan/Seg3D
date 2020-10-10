import easydict as ed
import os
##########################
# Root options-1
##########################
opt = ed.EasyDict()
opt.use_wave_lr = True
opt.is_train = True                         # The flag to determine whether it is train or not [True/False]
opt.continue_train = False                   # The flag to determine whether to continue training
opt.which_epoch = None                  # Which epoch to recover
opt.gpu_ids = [0]                           # gpu ids, [0/2/3] or their combination
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
##########################
# Data options
##########################
opt.data = ed.EasyDict()

opt.data.name = 'FOSSIL'                               # [MPI-main] the name of dataset [MPI, MIT] MPI-main
opt.data.fossil_name = ['XSC-03A', 'XSC-03B', 'XSC-16B']

opt.data.input_dim = 3*7                            # number of image channels [1/3], [3], [3*7] 上三张，下三张
opt.data.num_workers = 0                            # number of data loading threads
opt.data.use_norm = False  #True                            # whether use normalization [True/False]
opt.data.batch_size = 6                             # batch size [8]
opt.data.batch_size_test = 1
opt.data.load_size = 500                            # loading image size
opt.data.new_size = 288                             # the output of dataset's image size
opt.data.new_size_test = 256                        # [304] for test image size
opt.data.no_flip = False                            # the flag to determine not use random flip[True/False]
opt.data.unpaired = False
opt.data.serial_batches = True                      # take images in order or not [True/False], not shuffle while create
opt.data.is_train = opt.is_train                    # the flag to determine whether it is train or not
opt.data.data_root = '../../Data/'        # dataset folder location
opt.data.preprocess = 'resize_and_crop_refine_v2'             # pre-process[resize_and_crop/crop/scale_width/scale_width_and_crop] [resize_and_crop]
opt.data.image_mean = (0., 0., 0.)      # image mean value  (0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)
opt.data.image_std = (1., 1., 1.)       # image standard difference value  (0.2023, 0.1994, 0.2010)


##########################
# Trainer options
##########################
opt.train = ed.EasyDict()

opt.train.mode = 'v4-Sequence'                 #[v1] train generator mode [self-sup/none], [self-sup/cross-deconv/self-sup-ms]
opt.train.pool_size = 100                   # image pool size
opt.train.print_freq = 100                  # frequency of showing training results on console
opt.train.display_freq = 100                # frequency of showing training results in the visdom
opt.train.update_html_freq = 500            # frequency of showing training results in the web html
opt.train.save_latest_freq = 1000           # frequency of saving the latest results
opt.train.save_epoch_freq = 4               # the starting epoch count: epoch_count += save_freq
opt.train.n_iter = 100                      # iterations at starting learning rate
opt.train.n_iter_decay = 100                # iterations to linearly decay learning rate to zero

opt.train.save_train_img = True
opt.train.save_per_n_ep = 10  # 10
opt.train.save_per_n_ep_train = 10  # 10
opt.train.epoch_count = 1 if not opt.continue_train else 1+opt.which_epoch                  # the starting epoch count: epoch_count += save_freq
opt.train.total_ep = 150  # 250, MIT-50

##########################
# Model options
##########################
opt.model = ed.EasyDict()

# generator options
opt.model.gen = ed.EasyDict()

opt.model.gen.mode = 'direct'               # the mode of gen [fusion/minus/direct/Y-unet]
opt.model.gen.encoder_name = 'vgg19'        # the name of encoder [vgg11/vgg19]
opt.model.gen.feature_dim = 256             # number of dimension in input features, while decoding [256]
opt.model.gen.input_dim = opt.data.input_dim                 # number of dimension in input image
opt.model.gen.output_dim = 2              # number of dimension in background layer
opt.model.gen.dim = 64                      # number of filters in the bottommost layer, while decoding
opt.model.gen.mlp_dim = 256                 # number of filters in MLP
opt.model.gen.activ = 'lrelu'                # activation function [relu/lrelu/prelu/selu/tanh]
opt.model.gen.n_layers = 3                  # number of conv blocks in decoder [4]
opt.model.gen.pad_type = 'zero'          # padding type [zero/reflect]
opt.model.gen.norm = 'in'                   # normalization layer [none/bn/in/ln], before[in]
opt.model.gen.vgg_pretrained = False         # whether use pretrained vgg [True/False]
opt.model.gen.decoder_init = False
opt.model.gen.decoder_mode = 'Residual'            # [Basic/Residual], use plain decoder or ResNet-Dilation decoder

# discriminator options
##########################
# Root options-2
##########################
# The root dir for saving trained parameters and log information
opt.output_root = '../ckpoints' + '-' + opt.train.mode + '-' + opt.data.name
##########################
# Optimization options
##########################
opt.optim = ed.EasyDict()

opt.optim.max_iter = 1000000                # maximum number of training iterations
opt.optim.weight_decay = 0.0001             # weight decay
opt.optim.n_iter_decay = 50  # 30                # iterations to linearly decay learning rate to zero [100]
opt.optim.beta1 = 0.0                       # Adam parameter
opt.optim.beta2 = 0.9                       # Adam parameter
opt.optim.init = 'kaiming'                  # initialization [gaussian/kaiming/xavier/orthogonal]
opt.optim.lr_g = 0.0001                     # initial learning rate for generator, [0.0001]
opt.optim.lr_policy = 'step'                # learning rate scheduler
opt.optim.step_size = 6000                # how often to decay learning rate
opt.optim.gamma = 0.5                       # how much to decay learning rate
opt.optim.epoch_count = 1                   # the starting epoch count: epoch_count += save_freq

opt.optim.crossentropy_w = 1.0

##########################
# Logger options
##########################
opt.logger = ed.EasyDict()

opt.logger.image_save_iter = 10000          # How often do you want to save output images during training
opt.logger.image_display_iter = 100         # How often do you want to display output images during training
opt.logger.display_size = 16                # How many images do you want to display each time
opt.logger.snapshot_save_iter = 10000       # How often do you want to save trained models
opt.logger.log_iter = 10                    # How often do you want to log the training stats
opt.logger.log_dir = opt.output_root+'/log/'      # The log dir for saving train log and image information
opt.logger.root_dir = opt.output_root       # The root dir for logging
opt.logger.is_train = opt.is_train          # Copy the `is_train` flag
opt.logger.display_id = 1                   # Window id of the web display
opt.logger.display_single_pane_ncols = 0    # If positive, display all images in a single visdom with certain cols'
opt.logger.no_html = False                  # do not save intermediate training results to web
opt.logger.display_port = 8097              # visdom port of the web display
