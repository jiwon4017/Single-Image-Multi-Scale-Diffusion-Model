from SinDDM.functions import *
import math

from torch import nn
from einops import rearrange
from functools import partial
import torch.nn.functional as F
from torchvision import utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from text2live_util.clip_extractor import *
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from torchvision import transforms
from models.blip import *
from pathlib import Path


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )
    
    def forward(self,x):
        x = x.to(torch.float32)
        x_avg_pool = torch.mean(x,1).unsqueeze(1)
        x_max_pool = torch.max(x,1)[0].unsqueeze(1)
        attention = torch.cat((x_avg_pool,x_max_pool),dim=1)
        attention = self.conv(attention)
        return x * attention
    

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, context):
        attn_output, _ = self.attn(x, context, context)
        return attn_output


# building block modules
class SinDDMConvBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, text_emb_dim=None, mult=1 ,norm=True, attention=True, cross_attn_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if exists(time_emb_dim) else None

        self.text_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(text_emb_dim, text_emb_dim),
        ) if exists(text_emb_dim) else None

        self.time_reshape = nn.Conv2d(time_emb_dim, dim, 1) if exists(time_emb_dim) else None
        self.text_reshape = nn.Conv2d(text_emb_dim, dim, 1) if exists(text_emb_dim) else None
        self.ds_conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.attention = SpatialAttention() if attention else nn.Identity()
        
        # num_heads = 8
        # if dim % num_heads != 0:
        #     num_heads = 4
        # if dim % num_heads != 0:
        #     num_heads = 2
        # if dim % num_heads != 0:
        #     num_heads = 1

        num_heads = next(h for h in [8, 4, 2, 1] if dim % h == 0)

        self.cross_attn = CrossAttention(dim, num_heads=num_heads) if exists(cross_attn_dim) else None

    def forward(self, x, time_emb=None, similarity_text_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_condition = self.mlp(time_emb)
            time_condition = rearrange(time_condition, 'b c -> b c 1 1')
            time_condition = self.time_reshape(time_condition) if exists(self.time_reshape) else time_condition
            h = h + time_condition

        if exists(self.text_mlp):
            assert exists(similarity_text_emb), 'text emb must be passed in'
            text_condition = self.text_mlp(similarity_text_emb)
            text_condition = rearrange(text_condition, 'b c -> b c 1 1')
            text_condition = self.text_reshape(text_condition) if exists(self.text_reshape) else text_condition

            # Flatten h and text_condition for cross attention
            h_flat = h.flatten(2).transpose(1, 2)
            text_condition = text_condition.flatten(2).transpose(1, 2)
            h_flat = self.cross_attn(h_flat, text_condition)
            attn = h_flat.transpose(1, 2).view_as(h)
            h = h + attn
        h = self.net(h)
        return h + self.res_conv(x)


# denoiser model
class SinDDMNet(nn.Module):
    def __init__(
            self,
            dim,
            out_dim=None,
            channels=3,
            with_time_emb=True,
            multiscale=False,
            device=None
    ):
        super().__init__()
        self.device = device
        self.channels = channels
        self.multiscale = multiscale

        if with_time_emb:
            time_dim = 32

            if multiscale:
                self.SinEmbTime = SinusoidalPosEmb(time_dim)
                self.SinEmbScale = SinusoidalPosEmb(time_dim)
                self.time_mlp = nn.Sequential(
                    nn.Linear(time_dim * 2, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
            else:
                self.time_mlp = nn.Sequential(
                    SinusoidalPosEmb(time_dim),
                    nn.Linear(time_dim, time_dim * 4),
                    nn.GELU(),
                    nn.Linear(time_dim * 4, time_dim)
                )
        else:
            time_dim = None
            self.time_mlp = None

        text_dim = 512  # assuming the text embedding dimension is 512
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        half_dim = int(dim / 2)

        # U-Net architecture
        self.down1 = SinDDMConvBlock(channels, half_dim, time_emb_dim=time_dim, text_emb_dim=time_dim, cross_attn_dim=text_dim)
        self.down2 = SinDDMConvBlock(half_dim, dim, time_emb_dim=time_dim, text_emb_dim=time_dim, cross_attn_dim=text_dim)
        self.down3 = SinDDMConvBlock(dim, dim, time_emb_dim=time_dim, text_emb_dim=time_dim, cross_attn_dim=text_dim)

        self.up3 = SinDDMConvBlock(dim * 2, dim, time_emb_dim=time_dim, text_emb_dim=time_dim, cross_attn_dim=text_dim)  # dim*2 due to concatenation in U-Net
        self.up2 = SinDDMConvBlock(dim * 2, half_dim, time_emb_dim=time_dim, text_emb_dim=time_dim, cross_attn_dim=text_dim)
        self.up1 = SinDDMConvBlock(half_dim * 2, channels, time_emb_dim=time_dim, text_emb_dim=time_dim, cross_attn_dim=text_dim)

        out_dim = out_dim or channels
        self.final_conv = nn.Conv2d(channels, out_dim, 1)

    def forward(self, x, time, scale=None, similarity_text_emb=None):
        if self.multiscale:
            scale_tensor = torch.ones(size=time.shape).to(device=self.device) * scale
            t = self.SinEmbTime(time)
            s = self.SinEmbScale(scale_tensor)
            t_s_vec = torch.cat((t, s), dim=1)
            time_cond_vec = self.time_mlp(t_s_vec)
        else:
            t = self.time_mlp(time) if self.time_mlp else None
            time_cond_vec = t

        if similarity_text_emb is not None:
            similarity_text_emb = similarity_text_emb.float()  # Convert similarity_text_emb to float
            text_cond_vec = self.text_mlp(similarity_text_emb)

            text_cond_vec = text_cond_vec.unsqueeze(1).repeat(1, time_cond_vec.size(0) // text_cond_vec.size(0), 1).view_as(time_cond_vec)
        else:
            text_cond_vec = torch.zeros_like(time_cond_vec)

        # Encoder path
        d1 = self.down1(x, time_cond_vec, text_cond_vec)
        d2 = self.down2(d1, time_cond_vec, text_cond_vec)
        d3 = self.down3(d2, time_cond_vec, text_cond_vec)

        # Decoder path
        u3 = self.up3(torch.cat([d3, d3], dim=1), time_cond_vec, text_cond_vec)  # Skip connection (d3 to u3)
        u2 = self.up2(torch.cat([d2, u3], dim=1), time_cond_vec, text_cond_vec)  # Skip connection (d2 to u2)
        u1 = self.up1(torch.cat([d1, u2], dim=1), time_cond_vec, text_cond_vec)  # Skip connection (d1 to u1)
        return self.final_conv(u1)

    


class MultiScaleGaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            image, 
            blip_model, 
            clip_model,  
            *,
            save_interm=False,
            results_folder = '/Results',
            n_scales,
            scale_factor,
            image_sizes,
            scale_mul=(1, 1),
            channels=3,
            timesteps=100,
            train_full_t=False,
            scale_losses=None,
            loss_factor=1,
            loss_type='l1',
            betas=None,
            device=None,
            reblurring=True,
            sample_limited_t=False,
            omega=0,
            similarity_text_emb=None, 
            positive = None,
            total_steps=120001,
            negative = None,
    ):
        super().__init__()
        self.device = device
        self.save_interm = save_interm
        self.results_folder = Path(results_folder)
        self.channels = channels
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.image_sizes = ()
        self.scale_mul = scale_mul

        self.sample_limited_t = sample_limited_t
        self.reblurring = reblurring

        self.img_prev_upsample = None

        # CLIP guided sampling
        self.clip_guided_sampling = False
        self.guidance_sub_iters = None
        self.stop_guidance = None
        self.quantile = 0.8
        self.clip_model = None
        self.clip_strength = None
        self.clip_text = ''
        self.avg_text_embedds = None
        self.text_embedds = None
        self.text_embedds_hr = None
        self.text_embedds_lr = None
        self.clip_text_features = None
        self.clip_score = []
        self.clip_mask = None
        self.llambda = 0
        self.x_recon_prev = None

        # for clip_roi
        self.clip_roi_bb = []

        # omega tests
        self.omega = omega

        # ROI guided sampling
        self.roi_guided_sampling = False
        self.roi_bbs = []  # roi_bbs - list of [y,x,h,w]
        self.roi_bbs_stat = []  # roi_bbs_stat - list of [mean_tensor[1,3,1,1], std_tensor[1,3,1,1]]
        self.roi_target_patch = []

        for i in range(n_scales):  # flip xy->hw
            self.image_sizes += ((image_sizes[i][1], image_sizes[i][0]),)

        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.num_timesteps_trained = []
        self.num_timesteps_ideal = []
        self.num_timesteps_trained.append(self.num_timesteps)
        self.num_timesteps_ideal.append(self.num_timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        sigma_t = np.sqrt(1. - alphas_cumprod) / np.sqrt(alphas_cumprod) # sigma_t = sqrt_one_minus_alphas_cumprod_div_sqrt_alphas_cumprod

        # flag to force training of all the timesteps across all scales
        if scale_losses is not None:
            for i in range(n_scales - 1):
                self.num_timesteps_ideal.append(
                    int(np.argmax(sigma_t > loss_factor * scale_losses[i])))
                if train_full_t:
                    self.num_timesteps_trained.append(
                        int(timesteps))
                else:
                    self.num_timesteps_trained.append(self.num_timesteps_ideal[i+1])

        # gamma blur schedule
        gammas = torch.zeros(size=(n_scales - 1, self.num_timesteps), device=self.device)
        for i in range(n_scales - 1):
            gammas[i,:] = (torch.tensor(sigma_t, device=self.device) / (loss_factor * scale_losses[i])).clamp(min=0, max=1)

        self.register_buffer('gammas', gammas)

        self.total_steps = total_steps  # 총 step 수
        self.current_step = 0

        clip_cfg = {"clip_model_name": "ViT-B/32", "clip_affine_transform_fill": True, "n_aug": 16}
        self.clip_model = ClipExtractor(clip_cfg)

        # 평균 텍스트 임베딩 생성 및 저장
        self.similarity_text_emb = similarity_text_emb
        self.positive = positive
        self.negative = negative

    def roi_patch_modification(self, x_recon, scale=0, eta=0.8):
        x_modified = x_recon
        for bb in self.roi_bbs:
            bb = [int(bb_i / np.power(self.scale_factor, self.n_scales - scale - 1)) for bb_i in bb]
            bb_y, bb_x, bb_h, bb_w = bb
            target_patch_resize = F.interpolate(self.roi_target_patch[scale], size=(bb_h, bb_w))
            x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w] = eta * target_patch_resize + (1 - eta) * x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w]
        return x_modified

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, s, noise):
        x_recon_ddpm = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

        if not self.reblurring or s == 0:
            return x_recon_ddpm, x_recon_ddpm
        else:
            cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)
            x_tm1_mix = (x_recon_ddpm - extract(cur_gammas, t, x_recon_ddpm.shape) * self.img_prev_upsample) / (
                        1 - extract(cur_gammas, t, x_recon_ddpm.shape))
            x_t_mix = x_recon_ddpm
            return x_tm1_mix, x_t_mix

    def q_posterior(self, x_start, x_t_mix, x_t, t, s):
        if not self.reblurring or s == 0:
            posterior_mean = (
                    extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                    extract(self.posterior_mean_coef2, t, x_t.shape) * x_t

            )
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        elif t[0] > 0:
            x_tm1_mix = x_start
            posterior_variance_low = torch.zeros(x_t.shape,
                                                 device=self.device)
            posterior_variance_high = 1 - extract(self.alphas_cumprod, t - 1, x_t.shape)
            omega = self.omega
            posterior_variance = (1 - omega) * posterior_variance_low + omega * posterior_variance_high
            posterior_log_variance_clipped = torch.log(posterior_variance.clamp(1e-20, None))

            var_t = posterior_variance

            posterior_mean = extract(self.sqrt_alphas_cumprod, t - 1, x_t.shape) * x_tm1_mix + \
                                    torch.sqrt(1 - extract(self.alphas_cumprod, t - 1, x_t.shape) - var_t) * \
                                    (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t_mix) / \
                                    extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        else:
            posterior_mean = x_start
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.enable_grad()
    def p_mean_variance(self, x, t, s, clip_denoised: bool):
        pred_noise = self.denoise_fn(x, t, scale=s)
        x_recon, x_t_mix = self.predict_start_from_noise(x, t=t, s=s, noise=pred_noise)
        cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)

        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (x_recon.clamp(-1., 1.) + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'denoised_t-{t[0]:03}_s-{s}.png'),
                             nrow=4)
        if self.clip_guided_sampling and (self.stop_guidance <= t[0] or s < self.n_scales - 1) and self.guidance_sub_iters[s] > 0:
            if clip_denoised:
                x_recon.clamp_(-1., 1.)

            if self.clip_mask is not None:
                x_recon = x_recon * (1 - self.clip_mask) + (
                        (1 - self.llambda) * self.x_recon_prev + self.llambda * x_recon) * self.clip_mask
            x_recon.requires_grad_(True)

            x_recon_renorm = (x_recon + 1) * 0.5
            for i in range(self.guidance_sub_iters[s]):
                self.clip_model.zero_grad()
                if s > 0:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.avg_text_embedds)
                else:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.avg_text_embedds)

                clip_grad = torch.autograd.grad(score, x_recon, create_graph=False)[0]

                if self.clip_mask is None:
                    clip_grad, clip_mask = thresholded_grad(grad=clip_grad, quantile=self.quantile)
                    self.clip_mask = clip_mask.float()

                if self.save_interm:
                    final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
                    final_results_folder.mkdir(parents=True, exist_ok=True)
                    final_mask = self.clip_mask.type(torch.float64)

                    utils.save_image(final_mask,
                                     str(final_results_folder / f'clip_mask_s-{s}.png'),
                                     nrow=4)
                    utils.save_image((x_recon.clamp(-1., 1.) + 1) * 0.5,
                                     str(final_results_folder / f'clip_out_s-{s}_t-{t[0]}_subiter_{i}.png'),
                                     nrow=4)

                division_norm = torch.linalg.vector_norm(x_recon * self.clip_mask, dim=(1, 2, 3), keepdim=True) / torch.linalg.vector_norm(
                    clip_grad * self.clip_mask, dim=(1, 2, 3), keepdim=True)

                x_recon += self.clip_strength * division_norm * clip_grad * self.clip_mask
                x_recon.clamp_(-1., 1.)

                x_recon_renorm = (x_recon + 1) * 0.5
                self.clip_score.append(score.detach().cpu())

            self.x_recon_prev = x_recon.detach()

            plt.rcParams['figure.figsize'] = [16, 8]
            plt.plot(self.clip_score)
            plt.grid(True)
            plt.savefig(str(self.results_folder / 'clip_score'))
            plt.clf()

        elif self.roi_guided_sampling and (s < self.n_scales - 1):
            x_recon = self.roi_patch_modification(x_recon, scale=s)

        if int(s) > 0 and t[0] > 0 and self.reblurring:
            x_tm1_mix = extract(cur_gammas, t - 1, x_recon.shape) * self.img_prev_upsample + \
                        (1 - extract(cur_gammas, t - 1, x_recon.shape)) * x_recon
        else:
            x_tm1_mix = x_recon

        if clip_denoised:
            x_tm1_mix.clamp_(-1., 1.)
            x_t_mix.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_tm1_mix, x_t_mix=x_t_mix,
                                                                                  x_t=x, t=t, s=s)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, s, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, clip_denoised=clip_denoised)

        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        nonzero_mask_s = torch.tensor([True], device=self.device).float()

        return model_mean + nonzero_mask_s * nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, s):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'input_noise_s-{s}.png'),
                             nrow=4)
        if self.sample_limited_t and s < (self.n_scales - 1):
            t_min = self.num_timesteps_ideal[s + 1]
        else:
            t_min = 0
        for i in tqdm(reversed(range(t_min, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), s)
            if self.save_interm:
                final_img = (img + 1) * 0.5
                utils.save_image(final_img,
                                 str(final_results_folder / f'output_t-{i:03}_s-{s}.png'),
                                 nrow=4)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, scale_0_size=None, s=0):
        if scale_0_size is not None:
            image_size = scale_0_size
        else:
            image_size = self.image_sizes[0]
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size[0], image_size[1]), s=s)

    @torch.no_grad()
    def p_sample_via_scale_loop(self, batch_size, img, s, custom_t=None):
        device = self.betas.device
        if custom_t is None:
            total_t = self.num_timesteps_ideal[min(s, self.n_scales - 1)] - 1
        else:
            total_t = custom_t
        b = batch_size
        self.img_prev_upsample = img
        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'clean_input_s_{s}.png'),
                             nrow=4)
        img = self.q_sample(x_start=img, t=torch.Tensor.expand(torch.tensor(total_t, device=device), batch_size), noise=None)

        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img,
                             str(final_results_folder / f'noisy_input_s_{s}.png'),
                             nrow=4)

        if self.clip_mask is not None:
            if s > 0:
                mul_size = [int(self.image_sizes[s][0] * self.scale_mul[0]), int(self.image_sizes[s][1] * self.scale_mul[1])]
                self.clip_mask = F.interpolate(self.clip_mask, size=mul_size, mode='bilinear')
                self.x_recon_prev = F.interpolate(self.x_recon_prev, size=mul_size, mode='bilinear')
            else:
                self.clip_mask = None

        if self.sample_limited_t and s < (self.n_scales - 1):
            t_min = self.num_timesteps_ideal[s + 1]
        else:
            t_min = 0
        for i in tqdm(reversed(range(t_min, total_t)), desc='sampling loop time step', total=total_t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), s)
            if self.save_interm:
                final_img = (img + 1) * 0.5
                utils.save_image(final_img,
                                 str(final_results_folder / f'output_t-{i:03}_s-{s}.png'),
                                 nrow=4)
        return img

    @torch.no_grad()
    def sample_via_scale(self, batch_size, img, s, scale_mul=(1, 1), custom_sample=False, custom_img_size_idx=0, custom_t=None, custom_image_size=None):
        if custom_sample:
            if custom_img_size_idx >= self.n_scales:
                size = self.image_sizes[self.n_scales - 1]
                factor = self.scale_factor ** (custom_img_size_idx + 1 - self.n_scales)
                size = (int(size[0] * factor), int(size[1] * factor))
            else:
                size = self.image_sizes[custom_img_size_idx]
        else:
            size = self.image_sizes[s]
        image_size = (int(size[0] * scale_mul[0]), int(size[1] * scale_mul[1]))
        if custom_image_size is not None:
            image_size = custom_image_size

        img = F.interpolate(img, size=image_size, mode='bilinear')
        return self.p_sample_via_scale_loop(batch_size, img, s, custom_t=custom_t)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, s, noise=None, x_orig=None, similarity_text_emb=None,positive = None, negative = None):
        clip_cfg = {"clip_model_name": "ViT-B/32",
                    "clip_affine_transform_fill": True,
                    "n_aug": 16}
        t2l_clip_extractor = ClipExtractor(clip_cfg)
        clip_model = t2l_clip_extractor

        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        if int(s) > 0:
            cur_gammas = self.gammas[s - 1].reshape(-1)
            x_mix = extract(cur_gammas, t, x_start.shape) * x_start + \
                    (1 - extract(cur_gammas, t, x_start.shape)) * x_orig
            x_noisy = self.q_sample(x_start=x_mix, t=t, noise=noise)
            x_recon = self.denoise_fn(x_noisy, t, s,similarity_text_emb = similarity_text_emb)
        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(x_noisy, t, s,similarity_text_emb=similarity_text_emb)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        elif self.loss_type == 'l1_pred_img':
            if int(s) > 0:
                cur_gammas = self.gammas[s - 1].reshape(-1)
                if t[0] > 0:
                    x_mix_prev = extract(cur_gammas, t - 1, x_start.shape) * x_start + \
                                 (1 - extract(cur_gammas, t - 1, x_start.shape)) * x_orig
                else:
                    x_mix_prev = x_orig
            else:
                x_mix_prev = x_start
            loss = (x_mix_prev - x_recon).abs().mean()
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, s, *args, **kwargs):
        if int(s) > 0:
            x_orig = x[0]
            x_recon = x[1]
            b, c, h, w = x_orig.shape
            device = x_orig.device
            img_size = self.image_sizes[s]
            assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()

            if self.current_step>0 and self.current_step%500==0:
                pos = self.positive
                neg = self.negative

                clip_image_embedding = self.clip_model.encode_image(x_recon)

                if pos.shape[0] != clip_image_embedding.shape[0]:
                    pos = pos.repeat(clip_image_embedding.shape[0] // pos.shape[0], 1)

                if neg.shape[0] != clip_image_embedding.shape[0]:
                    neg = neg.repeat(clip_image_embedding.shape[0] // neg.shape[0], 1)

                pos_clip_loss_value = cosine_loss(clip_image_embedding, pos)
                neg_clip_loss_value = cosine_loss(clip_image_embedding, neg)

                return self.p_losses(x_recon, t, s, x_orig=x_orig, *args, **kwargs) + neg_clip_loss_value * 0.1 - pos_clip_loss_value * 0.1
            else:
                return self.p_losses(x_recon, t, s, x_orig=x_orig, *args, **kwargs)
        else:
            b, c, h, w = x[0].shape
            device = x[0].device
            img_size = self.image_sizes[s]
            assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()
            return self.p_losses(x[0], t, s, *args, **kwargs)

def cosine_loss(x, y, scaling=1.2):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    return scaling * (1 - F.cosine_similarity(x, y).mean())
