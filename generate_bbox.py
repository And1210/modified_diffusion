from denoising_diffusion_pytorch.bbox_denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os

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
    objective = 'pred_x0'
).cuda()

trainer = Trainer(
    diffusion,
    '/ingenuity_NAS/16amf8_nas/16amf8_mount/data/datasets/CityPersons_Seg/CityPersons/train_seg',
    train_batch_size = 16,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                        # turn on mixed precision
    results_folder = 'results_bbox'
)

PATH = '/ingenuity_NAS/16amf8_nas/16amf8_mount/data/datasets/CP_Seg_Gen_bbox_MOT/train_seg'
trainer.load('', True)
trainer.generate(10000, PATH)
