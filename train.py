import os
import argparse
import torch
from omegaconf import OmegaConf
from tools.dataloader import get_loaders
from autoencoder.autoencoder_vit import ViTAutoencoder 
from losses.perceptual import LPIPSWithDiscriminator
from utils import file_name, Logger
from einops import rearrange
from torch.cuda.amp import GradScaler, autocast
from utils import AverageMeter
from evals.eval import test_psnr, test_ifvd

import torch
import time


def vit_trainer(rank, model, opt, d_opt, criterion, train_loader, test_loader, fp, logger=None, predictioner=True):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if rank == 0:
        rootdir = logger.logdir

    device = torch.device('cuda', rank)
    
    losses = dict()
    losses['ae_loss'] = AverageMeter()
    losses['d_loss'] = AverageMeter()
    check = time.time()

    accum_iter = 6
    disc_opt = False

    if fp:
        scaler = GradScaler()
        scaler_d = GradScaler()

    model.train()
    disc_start = criterion.discriminator_iter_start
    
    for it, (x, p) in enumerate(train_loader):

        if it > 1000000:
            break
        batch_size = x.size(0)

        x = x.to(device)
        x = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w') # videos

        wanted_result = x
        if predictioner:
            p = p.to(device)
            p = rearrange(p / 127.5 - 1, 'b t c h w -> b c t h w') # videos
            wanted_result = p

        if not disc_opt:
            with autocast():
                x_tilde, vq_loss  = model(x)
                if it % accum_iter == 0:
                    model.zero_grad()
                ae_loss = criterion(vq_loss, wanted_result, 
                                    rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size),
                                    optimizer_idx=0,
                                    global_step=it)

                ae_loss = ae_loss / accum_iter
            
            scaler.scale(ae_loss).backward()

            if it % accum_iter == accum_iter - 1:
                scaler.step(opt)
                scaler.update()

            losses['ae_loss'].update(ae_loss.item(), 1)

        else:
            if it % accum_iter == 0:
                criterion.zero_grad()

            with autocast():
                with torch.no_grad():
                    x_tilde, vq_loss = model(x)
                d_loss = criterion(vq_loss, wanted_result, 
                         rearrange(x_tilde, '(b t) c h w -> b c t h w', b=batch_size),
                         optimizer_idx=1,
                         global_step=it)
                d_loss = d_loss / accum_iter
            
            scaler_d.scale(d_loss).backward()

            if it % accum_iter == accum_iter - 1:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler_d.unscale_(d_opt)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(criterion.discriminator_2d.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(criterion.discriminator_3d.parameters(), 1.0)

                scaler_d.step(d_opt)
                scaler_d.update()

            losses['d_loss'].update(d_loss.item() * 3, 1)

        if it % accum_iter == accum_iter - 1 and it // accum_iter >= disc_start:
            if disc_opt:
                disc_opt = False
            else:
                disc_opt = True

        if it % 100 == 0:
            log_('[Step %d][Time %.3f] [AELoss %f] [DLoss %f]' % (it, time.time() - check, losses['ae_loss'].average, losses['d_loss'].average))


        if it % 1000 == 0 and it > 0:
            fvd = test_ifvd(rank, model, test_loader, it, logger)
            psnr = test_psnr(rank, model, test_loader, it, logger)
            if logger is not None and rank == 0:
                logger.scalar_summary('train/ae_loss', losses['ae_loss'].average, it)
                logger.scalar_summary('train/d_loss', losses['d_loss'].average, it)
                logger.scalar_summary('test/psnr', psnr, it)
                logger.scalar_summary('test/fvd', fvd, it)

                log_('[Time %.3f] [AELoss %f] [DLoss %f] [PSNR %f]' %
                     (time.time() - check, losses['ae_loss'].average, losses['d_loss'].average, psnr))

                torch.save(model.state_dict(), rootdir + f'model_last.pth')
                torch.save(criterion.state_dict(), rootdir + f'loss_last.pth')
                torch.save(opt.state_dict(), rootdir + f'opt.pth')
                torch.save(d_opt.state_dict(), rootdir + f'd_opt.pth')
                torch.save(scaler.state_dict(), rootdir + f'scaler.pth')
                torch.save(scaler_d.state_dict(), rootdir + f'scaler_d.pth')

            losses = dict()
            losses['ae_loss'] = AverageMeter()
            losses['d_loss'] = AverageMeter()

        if it % 5000 == 0 and it > 0:
            torch.save(model.state_dict(), rootdir + f'model_{it}.pth')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='meteonet')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--config', type=str, default='configs/predictioner.yaml')
    parser.add_argument('--scale_lr', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--partial-load', action='store_true')
    parser.add_argument('--vit_sample', action='store_true', help='sample only from the vit')
    parser.add_argument('--res', type=int,help="Image resolution", default=256)
    
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    args.ddconfig   = config.model.params.ddconfig
    args.embed_dim  = config.model.params.embed_dim
    args.lossconfig = config.model.params.lossconfig
    args.lr         = config.model.base_learning_rate
    args.res        = config.model.params.ddconfig.resolution
    args.timesteps  = config.model.params.ddconfig.timesteps
    args.skip       = config.model.params.ddconfig.skip
    args.resume     = config.model.resume 
    args.amp        = config.model.amp
    args.predictioner = config.model.params.predictioner

    device = torch.device(args.device)

    """ ROOT DIRECTORY """
    fn = file_name(args)
    logger = Logger(fn)
    logger.log(args)
    logger.log(f'Log path: {logger.logdir}')
    rootdir = logger.logdir
    log_ = logger.log
    log_(f"Loading dataset {args.data_path} with resolution {args.res}")
    train_loader, test_loader, total_vid = get_loaders("METEO", args.res, args.timesteps, args.skip, args.batch_size, 1, 42, cond=False, data_location=args.data_path, predictioner=args.predictioner)
    log_(f"Generating model")

    model = ViTAutoencoder(args.embed_dim, args.ddconfig)

    if args.checkpoint:
        model_ckpt = torch.load(args.checkpoint)
        del model_ckpt["coords"]
        del model_ckpt["xy_pos_embedding"]
        del model_ckpt["encoder.to_patch_embedding.weight"]
        del model_ckpt["to_pixel.1.weight"]
        del model_ckpt["to_pixel.1.bias"]
        print("Partial loading")
        model.load_state_dict(model_ckpt, strict=not args.partial_load)
        del model_ckpt
    model = model.to(device)

    criterion = LPIPSWithDiscriminator(disc_start   = args.lossconfig.params.disc_start,
                                       timesteps    = args.ddconfig.timesteps).to(device)


    opt = torch.optim.AdamW(model.parameters(), 
                             lr=args.lr, 
                             betas=(0.5, 0.9)
                             )

    d_opt = torch.optim.AdamW(list(criterion.discriminator_2d.parameters()) + list(criterion.discriminator_3d.parameters()), 
                             lr=args.lr, 
                             betas=(0.5, 0.9))


    if args.vit_sample:
        model.eval()
        for batch in train_loader: 
            x, _ = batch
            x = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w') # videos
            x = x.to(device)
            with torch.no_grad():
                print("Sampling")
                model(x)
            print("Sampled")
        exit()

    torch.save(model.state_dict(), rootdir + f'net_init.pth')

    fp = args.amp
    rank = 0
    vit_trainer(rank, model, opt, d_opt, criterion, train_loader, test_loader, fp, logger, args.predictioner)

     
if __name__ == '__main__':
    main()