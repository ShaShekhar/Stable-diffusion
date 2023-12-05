import argparse, os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import islice
from torch import autocast
from contextlib import nullcontext

from stable_diffusion import *

c_params = { 
    'linear_start': 0.0015, 'linear_end': 0.0205, 'timesteps': 1000,
    'log_every_t': 100, 'loss_type': 'l1', 'first_stage_key': 'image',
    'monitor': 'val/loss', 'image_size': 64, 'channels': 4, 'scale_factor': 0.18215,
    'use_ema': False,

    'first_stage_config': {
        'embed_dim': 4, 'monitor': 'val/rec_loss', 
        'ddconfig': {
            'double_z': True, 'z_channels': 4,
            'in_channels': 3, 'out_ch': 3, 'ch_out': 128,
            'resolution': 512, 'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2, 'attn_resolutions': [],
            'dropout': 0.0
        },
        'lossconfig': {'torch.nn.Identity'}
    },
    'cond_stage_config': '__is_first_stage__',
    'cond_stage_trainable': False, 'cond_stage_key': 'masked_image', 'num_timesteps_cond': 1,
    'autoencoder': True,

    'unet_config': {
        'image_size': 64, 'in_channels': 9, 'out_channels': 4,
        'model_channels': 320, 'attention_resolutions': [4, 2, 1],
        'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4],
        'num_heads': 8, 'use_spatial_transformer': False,
        'use_checkpoint': True, 'legacy': False
    }
}

# For inpainting, the UNet has 5 additional input channels (4 for the encoded masked-image and 1 for
# the mask itself) whose weights were zero-initialized after restoring the non-inpainting checkpoint.

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0

    batch["mask"] = mask
    return batch

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--indir",
    #     type=str,
    #     nargs="?",
    #     help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    # )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/inpainting-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../sd-v1/sd-v1-5-inpainting.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    images = ["img.png"]
    masks = ["img_mask.png"]
    # masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    # images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    verbose = True
    # config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    # model = instantiate_from_config(config.model)
    ld_model = LatentDiffusion(**c_params)
    # for k, v in model.state_dict().items():
    #     print(f'{str(v.shape):30s}', k)
    pl_sd = torch.load(opt.ckpt)["state_dict"]
    m, u = ld_model.load_state_dict(pl_sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    ld_model = ld_model.cpu()
    ld_model = ld_model.eval()
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    ld_model = ld_model.to(device)

    sampler = DDIMSampler(ld_model)
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    os.makedirs(opt.outdir, exist_ok=True)

    # start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with ld_model.ema_scope():
                for image, mask in tqdm(zip(images, masks)):
                    outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                    batch = make_batch(image, mask, device=device)

                    img_latent = ld_model.get_first_stage_encoding(ld_model.encode_first_stage(batch["image"]))
                    # encode masked image and concat downsampled mask
                    masked_img_latent = ld_model.get_first_stage_encoding(ld_model.encode_first_stage(batch["masked_image"]))
                    img_mask = torch.nn.functional.interpolate(batch["mask"], size=img_latent.shape[-2:])
                    # map the encoded img to nth time steps
                    img_latent = sampler.stochastic_encode(img_latent, torch.tensor([opt.ddim_steps-1]).to(device))

                    cond = {'c_concat': [[img_mask], [masked_img_latent]]}

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f] # [4,64,64]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     conditioning=cond,
                                                     eta=opt.ddim_eta,
                                                     x_T=img_latent)

                x_samples_ddim = ld_model.decode_first_stage(samples_ddim)
                predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                image = torch.clamp((batch["image"]+1.0)/2.0, min=0.0, max=1.0)
                mask = batch["mask"]

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

if __name__ == "__main__":
    main()