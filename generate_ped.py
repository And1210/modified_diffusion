from denoising_diffusion_pytorch.ped_denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import numpy as np

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 100,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1',            # L1 or L2
    objective = 'pred_noise'
).cuda()

trainer = Trainer(
    diffusion,
    '/ingenuity_NAS/16amf8_nas/16amf8_mount/data/datasets/CityPersons_Seg/CityPersons/train_seg',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                        # turn on mixed precision
    results_folder = 'results_ped'
)

PATH = '/ingenuity_NAS/16amf8_nas/16amf8_mount/data/datasets/CP_Seg_Gen_mk10/train_seg'
trainer.load('', True)
#for i in range(0, 101):
#    cur_path = os.path.join(PATH, 'ped_{}'.format(i))
#    if (not os.path.exists(cur_path)):
#        os.mkdir(cur_path)
#    trainer.generate(10, cur_path, i)
trainer.generate(10000, PATH, len(os.listdir(PATH)))

