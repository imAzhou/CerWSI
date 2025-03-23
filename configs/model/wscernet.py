backbone_type = 'uni'
neck_type = 'identity'
classifier_type = 'wscer_mlc'

backbone_output_dim = [1024]
backbone_ckpt = 'checkpoints/uni.bin'

use_peft = 'FourierFT'   # None, lora, FourierFT
num_patches = 196
