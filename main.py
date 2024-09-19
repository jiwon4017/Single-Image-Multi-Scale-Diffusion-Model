import torch
import numpy as np
import argparse
import os
import torchvision
from SinDDM.functions import create_img_scales
from SinDDM.models import SinDDMNet, MultiScaleGaussianDiffusion
from SinDDM.trainer import MultiscaleTrainer
from text2live_util.clip_extractor import ClipExtractor
from models.blip import blip_decoder
from torchvision.transforms import transforms
from SinDDM.models import *
from PIL import Image


def load_image(image_path, image_size=384, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

def generate_multiple_captions(bmodel, image, device, num_prompts=10, min_length=5, max_length=15):
    captions = []
    with torch.no_grad():
        for _ in range(num_prompts):
            # Different seed for each prompt
            seed = torch.randint(0, 10000, (1,)).item()
            torch.manual_seed(seed)

            # Generate caption
            caption = bmodel.generate(
                image.to(device), 
                sample=True, 
                top_p=0.9, 
                max_length=max_length, 
                min_length=min_length
            )
            captions.append(caption[0])
    return captions

from collections import Counter
def get_most_common_word(captions):
    all_words = " ".join(captions).split()
    word_counts = Counter(all_words)
    most_common_word = word_counts.most_common(1)[0][0]
    return most_common_word

def filter_captions(captions, word_to_exclude):
    filtered_captions = []
    for caption in captions:
        filtered_caption = " ".join([word for word in caption.split() if word != word_to_exclude])
        filtered_captions.append(filtered_caption)
    return filtered_captions

def generate_average_text_embedding(image, blip_model, clip_model, device, num_prompts=10, min_length=5, max_length=15):
    blip_model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
    ])
    image = transform(image)

    captions = generate_multiple_captions(
        blip_model, 
        image, 
        device, 
        num_prompts=num_prompts, 
        min_length=min_length, 
        max_length=max_length
    )

    with open('blip_prompt_10.txt', 'w') as f:
        for caption in captions:
            f.write(caption + '\n')


    text_embeddings = []
    for caption in captions:
        template = ["A photo of a {}", "A picture of a {}"]
        text_embedding = clip_model.get_text_embedding(caption, template).to(device).half()
        text_embeddings.append(text_embedding)
    
    img_embedding = clip_model.get_image_embedding(image[0].detach().unsqueeze(0))
    

    caption_loss_pairs = []
    for caption, txt_embedding in zip(captions, text_embeddings):
        if txt_embedding.shape[0] != img_embedding.shape[0]:
            txt_embedding = txt_embedding.repeat(img_embedding.shape[0] // txt_embedding.shape[0], 1)

        clip_loss_value = cosine_loss(txt_embedding, img_embedding)
        caption_loss_pairs.append((caption, clip_loss_value.item()))

    caption_loss_pairs.sort(key=lambda x: x[1])
    sorted_captions = [pair[0] for pair in caption_loss_pairs]
    sorted_losses = [pair[1] for pair in caption_loss_pairs]

    positive_captions = sorted_captions[:5]
    negative_captions = sorted_captions[5:]

    most_common_word = get_most_common_word(negative_captions)
    filtered_negative_captions = filter_captions(negative_captions, most_common_word)

    positive__embeddings = []
    negative__embeddings = []

    for caption in positive_captions:
        template = ["A photo of a {}", "A picture of a {}"]
        positive_text_embedding = clip_model.get_text_embedding(caption, template).to(device).half()
        positive__embeddings.append(positive_text_embedding)

    for caption in filtered_negative_captions:
        template = ["A photo of a {}", "A picture of a {}"]
        negative_text_embedding = clip_model.get_text_embedding(caption, template).to(device).half()
        negative__embeddings.append(negative_text_embedding)

    negative_average_text_embedding = torch.mean(torch.stack(negative__embeddings), dim=0)
    positive_average_text_embedding = torch.mean(torch.stack(positive__embeddings), dim=0)
    similarity_text = positive_captions[0]
    similarity_text_emb = clip_model.get_text_embedding(similarity_text, template).to(device).half()
    with open('similarity_text.txt', 'w') as f:
        f.write(similarity_text)

    return positive_average_text_embedding,negative_average_text_embedding, similarity_text_emb



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scope", help='choose training scope.', default='forest')
    parser.add_argument("--mode", help='choose mode: train, sample, clip_content, clip_style_gen, clip_style_trans, clip_roi, harmonization, style_transfer, roi')
    parser.add_argument("--input_image", help='content image for style transfer or harmonization.', default='seascape_composite_dragon.png')
    parser.add_argument("--start_t_harm", help='starting T at last scale for harmonization', default=5, type=int)
    parser.add_argument("--start_t_style", help='starting T at last scale for style transfer', default=15, type=int)
    parser.add_argument("--harm_mask", help='harmonization mask.', default='seascape_mask_dragon.png')
    parser.add_argument("--clip_text", help='enter CLIP text.', default='Fire in the Forest')
    parser.add_argument("--fill_factor", help='Dictates relative amount of pixels to be changed. Should be between 0 and 1.', type=float)
    parser.add_argument("--strength", help='Dictates the relative strength of CLIPs gradients. Should be between 0 and 1.',  type=float)
    parser.add_argument("--roi_n_tar", help='Defines the number of target ROIs in the new image.', default=1, type=int)
    parser.add_argument("--dataset_folder", help='choose dataset folder.', default='./datasets/forest/')
    parser.add_argument("--image_name", help='choose image name.', default='forest.jpeg')
    parser.add_argument("--results_folder", help='choose results folder.', default='./results/')
    parser.add_argument("--dim", help='widest channel dimension for conv blocks.', default=160, type=int)
    parser.add_argument("--scale_factor", help='downscaling step for each scale.', default=1.411, type=float)
    parser.add_argument("--timesteps", help='total diffusion timesteps.', default=100, type=int)
    parser.add_argument("--train_batch_size", help='batch size during training.', default=8, type=int)
    parser.add_argument("--grad_accumulate", help='gradient accumulation (bigger batches).', default=1, type=int)
    parser.add_argument("--train_num_steps", help='total training steps.', default=120001, type=int)
    parser.add_argument("--save_and_sample_every", help='n. steps for checkpointing model.', default=10000, type=int)
    parser.add_argument("--avg_window", help='window size for averaging loss (visualization only).', default=100, type=int)
    parser.add_argument("--train_lr", help='starting lr.', default=1e-3, type=float)
    parser.add_argument("--sched_k_milestones", nargs="+", help='lr scheduler steps x 1000.', default=[20, 40, 70, 80, 90, 110], type=int)
    parser.add_argument("--load_milestone", help='load specific milestone.', default=0, type=int)
    parser.add_argument("--sample_batch_size", help='batch size during sampling.', default=8, type=int)
    parser.add_argument("--scale_mul", help='image size retargeting modifier.', nargs="+", default=[1, 1], type=float)
    parser.add_argument("--sample_t_list", nargs="+", help='Custom list of timesteps corresponding to each scale (except scale 0).', type=int)
    parser.add_argument("--device_num", help='use specific cuda device.', default=0, type=int)
    parser.add_argument("--sample_limited_t", help='limit t in each scale to stop at the start of the next scale', action='store_true')
    parser.add_argument("--omega", help='sigma=omega*max_sigma.', default=0, type=float)
    parser.add_argument("--loss_factor", help='ratio between MSE loss and starting diffusion step for each scale.', default=1, type=float)

    args = parser.parse_args()

    print('num devices: '+ str(torch.cuda.device_count()))
    device = f"cuda:{args.device_num}"
    scale_mul = (args.scale_mul[0], args.scale_mul[1])
    sched_milestones = [val * 1000 for val in args.sched_k_milestones]
    results_folder = args.results_folder + '/' + args.scope

    # Load input image for average text embedding
    input_image_path = os.path.join(args.dataset_folder, args.image_name)
    input_image = load_image(input_image_path, device=device)

    # Initialize BLIP model
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    blip_model = blip_decoder(pretrained=model_url, image_size=384, vit='base')

    # Initialize CLIP model
    clip_cfg = {"clip_model_name": "ViT-B/32", "clip_affine_transform_fill": True, "n_aug": 16}
    clip_model = ClipExtractor(clip_cfg)

    # Create image scales
    sizes, rescale_losses, scale_factor, n_scales = create_img_scales(args.dataset_folder, args.image_name, scale_factor=args.scale_factor, create=True, auto_scale=50000)

    positive, negative, similarity_text_emb = generate_average_text_embedding(input_image, blip_model, clip_model, device)

    # Initialize SinDDMNet
    model = SinDDMNet(dim=args.dim, multiscale=True, device=device)
    model.to(device)

    # Initialize MultiScaleGaussianDiffusion with image, blip_model, and clip_model
    ms_diffusion = MultiScaleGaussianDiffusion(
        denoise_fn=model,
        image=input_image,
        blip_model=blip_model,
        clip_model=clip_model,
        device=device,
        save_interm=False,
        results_folder=results_folder,
        n_scales=n_scales,
        scale_factor=scale_factor,
        image_sizes=sizes,
        scale_mul=scale_mul,
        channels=3,
        timesteps=args.timesteps,
        train_full_t=True,
        scale_losses=rescale_losses,
        loss_factor=args.loss_factor,
        loss_type='l1',
        betas=None,
        reblurring=True,
        sample_limited_t=args.sample_limited_t,
        omega=args.omega,
        similarity_text_emb = similarity_text_emb,
        positive = positive,
        negative = negative
    ).to(device)


    if args.sample_t_list is None:
        sample_t_list = ms_diffusion.num_timesteps_ideal[1:]
    else:
        sample_t_list = args.sample_t_list

    ScaleTrainer = MultiscaleTrainer(
        ms_diffusion,
        folder=args.dataset_folder,
        n_scales=n_scales,
        scale_factor=scale_factor,
        image_sizes=sizes,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.grad_accumulate,
        ema_decay=0.995,
        fp16=False,
        save_and_sample_every=args.save_and_sample_every,
        avg_window=args.avg_window,
        sched_milestones=sched_milestones,
        results_folder=results_folder,
        device=device,
        similarity_text_emb=similarity_text_emb,
        positive = positive,
        negative = negative
    )

    if args.load_milestone > 0:
        ScaleTrainer.load(milestone=args.load_milestone)

    if args.mode == 'train':
        ScaleTrainer.train()
        ScaleTrainer.sample_scales(scale_mul=(1, 1), custom_sample=True, image_name=args.image_name, batch_size=args.sample_batch_size, custom_t_list=sample_t_list)
    elif args.mode == 'sample':
        ScaleTrainer.sample_scales(scale_mul=scale_mul, custom_sample=True, image_name=args.image_name, batch_size=args.sample_batch_size, custom_t_list=sample_t_list, save_unbatched=True)
    elif args.mode == 'clip_content':
        text_input = args.clip_text
        guidance_sub_iters = [0] + [1] * (n_scales - 1)
        assert args.strength is not None and 0 <= args.strength <= 1, f"Strength value should be between 0 & 1. Got: {args.strength}"
        assert args.fill_factor is not None and 0 <= args.fill_factor <= 1, f"fill_factor value should be between 0 & 1. Got: {args.fill_factor}"
        strength = args.strength
        quantile = 1. - args.fill_factor
        llambda = 0.2
        stop_guidance = 3
        ScaleTrainer.ema_model.reblurring = False
        ScaleTrainer.clip_sampling(clip_model=clip_model, text_input=text_input, strength=strength, sample_batch_size=args.sample_batch_size, custom_t_list=sample_t_list, quantile=quantile, guidance_sub_iters=guidance_sub_iters, stop_guidance=stop_guidance, save_unbatched=True, scale_mul=scale_mul, llambda=llambda)
    elif args.mode == 'clip_style_trans' or args.mode == 'clip_style_gen':
        text_input = args.clip_text + ' Style'
        guidance_sub_iters = [0] * (n_scales - 1) + [1]
        strength = 0.3
        quantile = 0.0
        llambda = 0.05
        stop_guidance = 3
        start_noise = args.mode == 'clip_style_gen'
        image_name = args.image_name.rsplit(".", 1)[0] + '.png'
        ScaleTrainer.ema_model.reblurring = False
        ScaleTrainer.clip_sampling(clip_model=clip_model, text_input=text_input, strength=strength, sample_batch_size=args.sample_batch_size, custom_t_list=sample_t_list, quantile=quantile, guidance_sub_iters=guidance_sub_iters, stop_guidance=stop_guidance, save_unbatched=True, scale_mul=scale_mul, llambda=llambda, start_noise=start_noise, image_name=image_name)
    elif args.mode == 'clip_roi':
        text_input = args.clip_text
        strength = 0.1
        num_clip_iters = 100
        num_denoising_steps = 3
        dataset_folder = os.path.join(args.dataset_folder, f'scale_{n_scales - 1}/')
        image_name = args.image_name.rsplit(".", 1)[0] + '.png'
        import cv2
        image_to_select = cv2.imread(dataset_folder + image_name)
        roi = cv2.selectROI(image_to_select)
        roi_perm = [1, 0, 3, 2]
        roi = [roi[i] for i in roi_perm]
        ScaleTrainer.ema_model.reblurring = False
        ScaleTrainer.clip_roi_sampling(clip_model=clip_model, text_input=text_input, strength=strength, sample_batch_size=args.sample_batch_size, num_clip_iters=num_clip_iters, num_denoising_steps=num_denoising_steps, clip_roi_bb=roi, save_unbatched=True)
    elif args.mode == 'roi':
        import cv2
        image_path = os.path.join(args.dataset_folder, f'scale_{n_scales - 1}', args.image_name.rsplit(".", 1)[0] + '.png')
        image_to_select = cv2.imread(image_path)
        roi = cv2.selectROI(image_to_select)
        image_to_select = cv2.cvtColor(image_to_select, cv2.COLOR_BGR2RGB)
        roi_perm = [1, 0, 3, 2]
        target_roi = [roi[i] for i in roi_perm]
        tar_y, tar_x, tar_h, tar_w = target_roi
        roi_bb_list = []
        n_targets = args.roi_n_tar
        target_h = int(image_to_select.shape[0] * scale_mul[0])
        target_w = int(image_to_select.shape[1] * scale_mul[1])
        empty_image = np.ones((target_h, target_w, 3))
        target_patch_tensor = torchvision.transforms.ToTensor()(image_to_select[tar_y:tar_y + tar_h, tar_x:tar_x + tar_w, :])
        for i in range(n_targets):
            roi = cv2.selectROI(empty_image)
            roi_reordered = [roi[i] for i in roi_perm]
            roi_bb_list.append(roi_reordered)
            y, x, h, w = roi_reordered
            target_patch_tensor_resize = torch.nn.functional.interpolate(target_patch_tensor[None, :, :, :], size=(h, w))
            empty_image[y:y + h, x:x + w, :] = target_patch_tensor_resize[0].permute(1, 2, 0).numpy()
        empty_image = torchvision.transforms.ToTensor()(empty_image)
        torchvision.utils.save_image(empty_image, os.path.join(args.results_folder, args.scope, f'roi_patches.png'))
        ScaleTrainer.roi_guided_sampling(custom_t_list=sample_t_list, target_roi=target_roi, roi_bb_list=roi_bb_list, save_unbatched=True, batch_size=args.sample_batch_size, scale_mul=scale_mul)
    elif args.mode == 'style_transfer' or args.mode == 'harmonization':
        i2i_folder = os.path.join(args.dataset_folder, 'i2i')
        if args.mode == 'style_transfer':
            start_s = n_scales - 1
            start_t = args.start_t_style
            use_hist = True
        else:
            start_s = n_scales - 1
            start_t = args.start_t_harm
            use_hist = False
        custom_t = [0] * (n_scales - 1) + [start_t]
        hist_ref_path = f'{args.dataset_folder}scale_{start_s}/'
        ScaleTrainer.ema_model.reblurring = True
        ScaleTrainer.image2image(input_folder=i2i_folder, input_file=args.input_image, mask=args.harm_mask, hist_ref_path=hist_ref_path, batch_size=args.sample_batch_size, image_name=args.image_name, start_s=start_s, custom_t=custom_t, scale_mul=(1, 1), device=device, use_hist=use_hist, save_unbatched=True, auto_scale=50000, mode=args.mode)
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    main()
    quit()
