import argparse, os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import islice
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from stable_diffusion import *

c_params = {
    'linear_start': 0.0015, 'linear_end': 0.0205, 'log_every_t': 100, 
    'timesteps': 1000, 'loss_type': 'l1', 'first_stage_key': 'image',
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
    'cond_stage_config': {'target': 'ldm.modules.encoders.modules.FrozenCLIPEmbedder'},
    'cond_stage_trainable': False, 'cond_stage_key': 'txt', 'num_timesteps_cond': 1,
    'conditioning_key': 'hybrid', 'autoencoder': True,

    'unet_config': {
        'image_size': 64, 'in_channels': 9, 'out_channels': 4,
        'model_channels': 320, 'attention_resolutions': [4, 2, 1],
        'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4],
        'num_heads': 8, 'use_spatial_transformer': True,
        'transformer_depth': 1, 'context_dim': 768, 
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

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
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
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements."
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
    # parser.add_argument(
    #     "--n_iter",
    #     type=int,
    #     default=1,
    #     help="sample this often",
    # )
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
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../sd-v1/sd-v1-5-inpainting.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
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
    model = LatentDiffusion(**c_params)
    # for k, v in model.state_dict().items():
    #     print(f'{str(v.shape):30s}', k)
    pl_sd = torch.load(opt.ckpt)["state_dict"]
    m, u = model.load_state_dict(pl_sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model = model.cpu()
    model = model.eval()

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    os.makedirs(opt.outdir, exist_ok=True)
    # outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    # sample_path = os.path.join(outpath, "samples")
    # os.makedirs(sample_path, exist_ok=True)
    # base_count = len(os.listdir(sample_path))
    # grid_count = len(os.listdir(outpath)) - 1

    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in tqdm(data, desc="data"):
                    for image, mask in tqdm(zip(images, masks)):
                        outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                        batch = make_batch(image, mask, device=device)

                        img_latent = model.get_first_stage_encoding(model.encode_first_stage(batch["image"]))
                        # encode masked image and concat downsampled mask
                        masked_img_latent = model.get_first_stage_encoding(model.encode_first_stage(batch["masked_image"]))
                        masked_img_latent = torch.cat([masked_img_latent]*2)
                        img_mask = torch.nn.functional.interpolate(batch["mask"], size=img_latent.shape[-2:])
                        img_mask = torch.cat([img_mask]*2)

                        img_latent = sampler.stochastic_encode(img_latent, torch.tensor([opt.ddim_steps-1]).to(device))

                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        # print(f"prompt shape: {c.shape}") [1,77,768]
                        c_in = torch.cat([uc, c])

                        cond = {'c_concat': [[img_mask], [masked_img_latent]], 'c_crossattn': [c_in]}

                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f] # [4,64,64]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         conditioning=cond,
                                                         eta=opt.ddim_eta,
                                                         #mask=img_mask,
                                                         #x0=img_latent,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         x_T=img_latent)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    image = torch.clamp((batch["image"]+1.0)/2.0, min=0.0, max=1.0)
                    mask = batch["mask"]

                    inpainted = (1-mask)*image+mask*predicted_image
                    inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255

                    # pred_img = predicted_image.cpu().numpy().transpose(0,2,3,1)[0]*255
                    # pred_img_path = os.path.join(opt.outdir, 'pred.png')
                    # Image.fromarray(pred_img.astype(np.uint8)).save(pred_img_path)

                    Image.fromarray(inpainted.astype(np.uint8)).save(outpath)

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

if __name__ == "__main__":
    main()