from denoising_diffusion_pytorch.shanghai_denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

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
    '/ingenuity_NAS/16amf8_nas/16amf8_mount/data/datasets/Shanghai/part_A_final/train_data',
    train_batch_size = 2,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                        # turn on mixed precision
    results_folder = 'results_shanghai'
)

#trainer.load('', True)
trainer.train()