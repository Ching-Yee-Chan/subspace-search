import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import argparse
import numpy as np

from PyQt5 import QtWidgets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='desired dataset', choices=['MNIST', 'PGGAN_celebahq', 'GANSynth', 'IMGAN_flight', 'IMGAN_chair', 'IM-AE'])
    parser.add_argument('--output_path', help='where the saved results would be put', default='.')
    parser.add_argument('--initial', help='intial file')
    parser.add_argument('--use_approx', help='Use stochastic jacobian', default=True)
    args = parser.parse_args()

    init_latent = None
    if args.initial:
        fr = open(args.initial, 'r')
        latent_size = int(fr.readline())

        init_latent = []
        line = fr.readline()
        latent = line.split(' ')
        for j in range(latent_size):
            init_latent.append(float(latent[j]))
        init_latent = np.array(init_latent)

    dataset = args.dataset
    output_path = args.output_path
    title = ""
    if dataset == 'MNIST':
        from SearchUI import ImageSearchUI
        from models.MNISTGenerator import MNISTGenerator

        model = MNISTGenerator(args.use_approx)
        weights_path = '../pretrained_weights/MNIST/model.ckpt-155421'
        model.load_model(weights_path)
        title = "MNIST"
    elif dataset == 'PGGAN_celebahq':
        from SearchUI import ImageSearchUI
        from models.PGGANWrapper import PGGANWrapper

        weights_path = '../pretrained_weights/PGGAN/karras2018iclr-celebahq-1024x1024.pkl'
        model = PGGANWrapper(weights_path, args.use_approx)
        title = "PG-GAN"
    elif dataset == 'GANSynth':
        from SearchUI import AudioSearchUI
        from models.GANSynthWrapper import GANSynthWrapper

        model = GANSynthWrapper('../pretrained_weights/GANSynth/acoustic_only', 16000, args.use_approx)
        title = "GANSynth"
    elif dataset == 'IMGAN_flight':
        from SearchUI import OpenGLSearchUI
        from models.IMGAN import IMGAN

        model = IMGAN(args.use_approx)
        weights_path1 = '../pretrained_weights/IMGAN_flight/02691156_vox128_z_128_128/ZGAN.model-10000'
        weights_path2 = '../pretrained_weights/IMGAN_flight/02691156_vox128_64/IMAE.model-194'
        model.load_model(weights_path1, weights_path2)
        title = "IM-GAN"
    elif dataset == 'IMGAN_chair':
        from SearchUI import OpenGLSearchUI
        from models.IMGAN import IMGAN

        model = IMGAN(args.use_approx)
        weights_path1 = '../pretrained_weights/IMGAN_chair/03001627_vox_z_128_128/ZGAN.model-10000'
        weights_path2 = '../pretrained_weights/IMGAN_chair/03001627_vox_64/IMAE.model-191'
        model.load_model(weights_path1, weights_path2)
        title = "IM-GAN"
    elif dataset == "IM-AE":
        from SearchUI import OpenGLSearchUI
        from models.IM_AE import IM_AE
        # python main.py --ae --sample_dir samples/im_ae_out --start 0 --end 9000 --getz
        parser = argparse.ArgumentParser()
        parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
        parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")
        parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float, help="Learning rate for adam [0.00005]")
        parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
        parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img", help="The name of dataset")
        parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="../checkpoint", help="Directory name to save the checkpoints [checkpoint]")
        parser.add_argument("--data_dir", action="store", dest="data_dir", default="../data/all_vol256_img/", help="Root directory of dataset [data]")
        parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="../samples/im_ae_out", help="Directory name to save the image samples [samples]")
        parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int, help="Voxel resolution for coarse-to-fine training [64]")
        parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
        parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
        parser.add_argument("--end", action="store", dest="end", default=9000, type=int, help="In testing, output shapes [start:end]")
        parser.add_argument("--ae", action="store_true", dest="ae", default=True, help="True for ae [False]")
        parser.add_argument("--svr", action="store_true", dest="svr", default=False, help="True for svr [False]")
        parser.add_argument("--getz", action="store_true", dest="getz", default=True, help="True for getting latent codes [False]")
        parser.add_argument("--use_approx", action="store_true", dest="use_approx", default=args.use_approx, help="True for using approximation")
        FLAGS = parser.parse_known_args()[0]
        model = IM_AE(FLAGS)
        title = "IM-GAN"

    app = QtWidgets.QApplication(sys.argv)
    if dataset == 'MNIST' or dataset == 'PGGAN_celebahq':
        ui_window = ImageSearchUI(model, output_path)
    elif dataset == 'GANSynth':
        ui_window = AudioSearchUI(model, output_path, 16000)
    elif dataset == 'IMGAN_flight' or dataset == 'IMGAN_chair' or dataset == 'IM-AE':
        ui_window = OpenGLSearchUI(model, output_path)
    # ui_window.setWindowTitle(title)
    ui_window.setWindowTitle("Differential Subspace Search")

    ui_window.start_search(init_latent)
    app.exec_()
    sys.exit(0)