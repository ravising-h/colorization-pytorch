import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms

from util import util


class Colorization_Dataset(Dataset):
    """Colorization dataset."""

    def __init__(self,  root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        if isinstance(self.root_dir, str):
            self.root_dir = [self.root_dir]
        self.transform = transform
        self.path_to_images = list()
        for path in self.root_dir:
            fetch_full_path = lambda img_name,path: os.path.join(path, img_name) 
            self.path_to_images += list(map(fetch_full_path, ( os.listdir(path))))

    def __len__(self):
        return len(self.path_to_images)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx

        img_name = self.path_to_images[idx]
        image = PIL.open(img_name)
        
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    opt = TrainOptions().parse()

    opt.dataroot = './dataset/ilsvrc2012/%s/' % opt.phase
    colorization_dataset = Colorization_Dataset(opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.RandomChoice([transforms.Resize(opt.loadSize, interpolation=1),
                                                                            transforms.Resize(opt.loadSize, interpolation=2),
                                                                            transforms.Resize(opt.loadSize, interpolation=3),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=1),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=2),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=3)]),
                                                   transforms.RandomChoice([transforms.RandomResizedCrop(opt.fineSize, interpolation=1),
                                                                            transforms.RandomResizedCrop(opt.fineSize, interpolation=2),
                                                                            transforms.RandomResizedCrop(opt.fineSize, interpolation=3)]),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()]))
    dataset_loader = torch.utils.data.DataLoader(colorization_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads))

    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    model.print_networks(True)

    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # for i, data in enumerate(dataset):
        for i, data_raw in enumerate(dataset_loader):
            data_raw = data_raw.cuda()
            data = util.get_colorization_data(data_raw, opt, p=opt.sample_p)
            if(data is None):
                continue

            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                # time to load data
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                # time to do forward&backward
                t = time.time() - iter_start_time
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
