import torch.optim as optim
from net.network import VSSBJSCC
from data.datasets import get_loader
from utils import *
from net.model_names import MODEL_CHOICES, normalize_model_name

# torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import torchvision

parser = argparse.ArgumentParser(description='VSSBJSCC')
parser.add_argument('--training', action='store_true', help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K', choices=['CIFAR10', 'DIV2K'], help='train dataset name')
# parser.add_argument('--testset', type=str, default='ffhq', choices=['kodak', 'CLIC21', 'ffhq'], help='specify the testset for HR models')
parser.add_argument('--testset', type=str, default='kodak', choices=['kodak', 'CLIC21', 'ffhq'], help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE', choices=['MSE', 'MS-SSIM'], help='evaluation metrics')
parser.add_argument('--model',
                    type=str,
                    default='VSSBJSCC_w/_SAandRA',
                    choices=MODEL_CHOICES,
                    help='VSSBJSCC model variant; legacy SwinJSCC_* names are also accepted')
parser.add_argument('--channel-type', type=str, default='awgn', choices=['awgn', 'rayleigh'], help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=str, default='192', help='bottleneck dimension')  # choices=['32','64','96','128','192']
parser.add_argument('--multiple-snr', type=str, default='13', help='random or fixed snr') # choices=['1','4','7','10','13']
parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'], help='VSSBJSCC model size')
parser.add_argument('--use-diffusion', action='store_true', help='enable lightweight DPM-Solver diffusion refiner')
parser.add_argument('--train-diffusion-only', action='store_true', help='freeze VSSBJSCC and train only diffusion refiner')
parser.add_argument('--finetune-decoder-diffusion', action='store_true', help='freeze encoder and fine-tune decoder + diffusion refiner')
parser.add_argument('--diffusion-steps', type=int, default=1, help='diffusion inference steps; safe low-noise refine uses one denoise step')
parser.add_argument('--diffusion-train-timesteps', type=int, default=256, help='diffusion training timesteps')
parser.add_argument('--diffusion-blend', type=float, default=0.01, help='conservative blend between decoder output and diffusion output')
parser.add_argument('--diffusion-base-channels', type=int, default=16, help='base channels of lightweight diffusion UNet')
parser.add_argument('--diffusion-strength', type=float, default=0.05, help='low-noise strength for safe diffusion refinement')
parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint to load for testing or fine-tuning')
parser.add_argument('--run-id', type=str, default='2026-05-24 00:51:24', help='history run id used when --checkpoint is not given')
parser.add_argument('--epoch-id', type=int, default=500, help='checkpoint epoch used with --run-id when --checkpoint is not given')
parser.add_argument('--no-preload', action='store_true', help='do not load a checkpoint; useful for training from scratch')
args = parser.parse_args()
args.model = normalize_model_name(args.model)


class config():
    seed = 42
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 500

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        train_data_dir = "/media/D/Dataset/CIFAR10/"
        test_data_dir = "/media/D/Dataset/CIFAR10/"
        batch_size = 128
        downsample = 2
        channel_number = int(args.C)
        encoder_kwargs = dict(
            model=args.model,
            img_size=(image_dims[1], image_dims[2]),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 256],
            depths=[2, 4],
            num_heads=[4, 8],
            C=channel_number,
            window_size=2,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        decoder_kwargs = dict(
            model=args.model,
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128],
            depths=[4, 2],
            num_heads=[8, 4],
            C=channel_number,
            window_size=2,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 5
        image_dims = (3, 256, 256)
        base_path = "/home/ippnet/Andrew/datasets/DIV2K"
        if args.testset == 'kodak':
            test_data_dir = ["/home/ippnet/Andrew/datasets/Kodak"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["/media/D/Dataset/HR_Image_dataset/clic2021/test/"]
        elif args.testset == 'ffhq':
            test_data_dir = ["/media/D/yangke/VSSBJSCC/data/ffhq/"]

        train_data_dir = [
            base_path + '/DIV2K_train_HR'
        ]
        batch_size = 8
        downsample = 4
        if args.model == 'VSSBJSCC_w/o_SAandRA' or args.model == 'VSSBJSCC_w/_SA':
            channel_number = int(args.C)
        else:
            channel_number = None

        if args.model_size == 'small':
            encoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 2, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 2, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
        elif args.model_size == 'base':
            encoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 6, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 6, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
        elif args.model_size == 'large':
            encoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=[128, 192, 256, 320],
                depths=[2, 2, 18, 2],
                num_heads=[4, 6, 8, 10],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )
            decoder_kwargs = dict(
                model=args.model,
                img_size=(image_dims[1], image_dims[2]),
                embed_dims=[320, 256, 192, 128],
                depths=[2, 18, 2, 2],
                num_heads=[10, 8, 6, 4],
                C=channel_number,
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
            )


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()


def load_weights(model_path, strict=True):
    pretrained = torch.load(model_path, map_location="cpu")
    msg = net.load_state_dict(pretrained, strict=strict)
    logger.info(f"Loaded checkpoint: {model_path}")
    logger.info(f"Load strict={strict}; message: {msg}")
    del pretrained


def configure_training_mode():
    if args.use_diffusion and args.train_diffusion_only:
        net.freeze_jscc()
    elif args.use_diffusion and args.finetune_decoder_diffusion:
        net.unfreeze_decoder_only()
    else:
        net.train()


def run_training_forward(input_image):
    if args.use_diffusion and args.train_diffusion_only:
        loss, recon_image, CBR, SNR, mse, loss_G = net.diffusion_training_loss(
            input_image,
            detach_jscc=True,
        )
        return loss, recon_image, CBR, SNR, mse, loss_G

    if args.use_diffusion and args.finetune_decoder_diffusion:
        loss_diff, recon_image, CBR, SNR, mse, loss_G = net.diffusion_training_loss(
            input_image,
            detach_jscc=False,
        )
        # Keep decoder reconstruction stable while learning the refiner.
        loss = loss_diff + 0.1 * loss_G
        return loss, recon_image, CBR, SNR, mse, loss_G

    recon_image, CBR, SNR, mse, loss_G = net(input_image)
    loss = loss_G
    return loss, recon_image, CBR, SNR, mse, loss_G


def train_one_epoch(args):
    configure_training_mode()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    if args.trainset == 'CIFAR10':
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            loss, recon_image, CBR, SNR, mse, loss_G = run_training_forward(input)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    else:
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.cuda()
            loss, recon_image, CBR, SNR, mse, loss_G = run_training_forward(input)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()


def test():
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    channel_number = args.C.split(",")
    for i in range(len(channel_number)):
        channel_number[i] = int(channel_number[i])
    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))
    for i, SNR in enumerate(multiple_snr):
        for j, rate in enumerate(channel_number):
            with torch.no_grad():
                if args.trainset == 'CIFAR10':
                    for batch_idx, (input, label) in enumerate(test_loader):
                        start_time = time.time()
                        input = input.cuda()
                        recon_image, CBR, SNR, mse, loss_G = net(input, SNR, rate)

                        elapsed.update(time.time() - start_time)
                        cbrs.update(CBR)
                        snrs.update(SNR)
                        if mse.item() > 0:
                            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                            psnrs.update(psnr.item())
                            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                            msssims.update(msssim)

                        log = (' | '.join([
                            f'Time {elapsed.val:.3f}',
                            f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                            f'SNR {snrs.val:.1f}',
                            f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                            f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                            f'Lr {cur_lr}',
                        ]))
                        logger.info(log)
                else:
                    for batch_idx, batch in enumerate(test_loader):
                        input, names = batch
                        start_time = time.time()
                        input = input.cuda()
                        recon_image, CBR, SNR, mse, loss_G = net(input, SNR, rate)
                        # torchvision.utils.save_image(recon_image, os.path.join("/media/D/yangke/VSSBJSCC/data/", f"recon/{names[0]}"))
                        elapsed.update(time.time() - start_time)
                        cbrs.update(CBR)
                        snrs.update(SNR)
                        if mse.item() > 0:
                            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                            psnrs.update(psnr.item())
                            msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                            msssims.update(msssim)
                            MSSSIM = -10 * np.log10(1 - msssim)
                        log = (' | '.join([
                            f'Time {elapsed.val:.3f}',
                            f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                            f'SNR {snrs.val:.1f}',
                            f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                            f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                            f'Lr {cur_lr}',
                        ]))
                        logger.info(log)
            results_snr[i, j] = snrs.avg
            results_cbr[i, j] = cbrs.avg
            results_psnr[i, j] = psnrs.avg
            results_msssim[i, j] = msssims.avg
            for t in metrics:
                t.clear()

    print("SNR: {}".format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}".format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params


def log_model_parameters(model, logger=None):
    total_params, trainable_params, non_trainable_params = count_parameters(model)

    msg = (
        f"Model parameters:\n"
        f"  Total params:        {total_params:,} ({total_params / 1e6:.3f} M)\n"
        f"  Trainable params:    {trainable_params:,} ({trainable_params / 1e6:.3f} M)\n"
        f"  Non-trainable params:{non_trainable_params:,} ({non_trainable_params / 1e6:.3f} M)"
    )

    if logger is not None:
        logger.info(msg)
    else:
        print(msg)



if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = VSSBJSCC(args, config)
    # Checkpoint loading is explicit for reproducibility.
    # If --checkpoint is omitted, the legacy history path is built from --run-id/--epoch-id.
    # Use --no-preload only when training from scratch.
    if args.no_preload:
        logger.info("No checkpoint loaded because --no-preload was specified.")
    else:
        if args.checkpoint is not None:
            model_path = args.checkpoint
        else:
            model_path = f"./history/{args.run_id}/models/{args.run_id}_EP{args.epoch_id}.model"

        # Old VSSBJSCC checkpoints do not contain diffusion_refiner.* weights.
        # Therefore, strict=False is required when --use-diffusion is enabled.
        load_weights(model_path, strict=not args.use_diffusion)

    net = net.cuda()

    if args.use_diffusion and args.train_diffusion_only:
        net.freeze_jscc()
        model_params = [{'params': net.diffusion_refiner.parameters(), 'lr': 2e-4}]
        cur_lr = 2e-4
        logger.info("Training phase: diffusion-only; VSSBJSCC encoder/decoder are frozen.")

    elif args.use_diffusion and args.finetune_decoder_diffusion:
        net.unfreeze_decoder_only()
        model_params = [
            {'params': net.decoder.parameters(), 'lr': 1e-6},
            {'params': net.diffusion_refiner.parameters(), 'lr': 1e-5},
        ]
        cur_lr = 1e-5
        logger.info("Training phase: fine-tune decoder + diffusion; encoder is frozen.")

    else:
        model_params = [{'params': net.parameters(), 'lr': config.learning_rate}]
        cur_lr = config.learning_rate
        logger.info("Training phase: standard VSSBJSCC.")

    log_model_parameters(net, logger)

    train_loader, test_loader = get_loader(args, config)
    optimizer = optim.Adam(model_params)
    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    if args.training:
        for epoch in range(steps_epoch, config.tot_epoch):
            train_one_epoch(args)
            save_freq = 1 if (args.use_diffusion and (args.train_diffusion_only or args.finetune_decoder_diffusion)) else config.save_model_freq
            if (epoch + 1) % save_freq == 0:
                save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
                test()
    else:
        test()
