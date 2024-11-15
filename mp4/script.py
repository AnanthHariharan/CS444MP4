import torch
from absl import app, flags
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from finetune import ViTLinear, inference, Trainer
import yaml
from datasets import get_flower102

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_name', 'vit_linear', 'Experiment name for corresponding hyperparameters in config.yaml')
flags.DEFINE_string('output_dir', 'run1', 'Output directory for logs and model checkpoints')
flags.DEFINE_string('encoder', 'vit_b_32', 'Encoder model to use (options: linear, resnet18, resnet50, vit_b_16, vit_b_32)')
flags.DEFINE_string('data_dir', './flower-dataset-reduced', 'Directory containing the Flower102 dataset')

def get_config(exp_name, encoder):
    dir_name = f'{FLAGS.output_dir}/runs-{encoder}-flower102/demo-{exp_name}'

    encoder_registry = {
        'ViTLinear': ViTLinear,
    }

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[exp_name]

    lr = config['lr']
    wd = config['wd']
    epochs = config['epochs']
    optimizer = config['optimizer']
    scheduler = config['scheduler']
    momentum = config['momentum']
    net_class = encoder_registry[config['net_class']]
    batch_size = config['batch_size']

    return net_class, dir_name, (optimizer, lr, wd, momentum), (scheduler, epochs), batch_size

def main(_):
    torch.set_num_threads(2)
    torch.manual_seed(0)

    print(f"Starting experiment: {FLAGS.exp_name} with encoder: {FLAGS.encoder}")

    net_class, dir_name, \
        (optimizer, lr, wd, momentum), \
        (scheduler, epochs), batch_size = \
        get_config(FLAGS.exp_name, FLAGS.encoder)

    train_data = get_flower102(FLAGS.data_dir, 'train')
    val_data = get_flower102(FLAGS.data_dir, 'val')
    test_data = get_flower102(FLAGS.data_dir, 'test')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    tmp_file_name = f'{dir_name}/best_model.pth'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(log_dir=f'{dir_name}/lr{lr:0.6f}_wd{wd:0.6f}', flush_secs=10)
    
    model = net_class(num_classes=102, encoder=FLAGS.encoder)
    model.to(device)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        writer=writer,
        optimizer=optimizer,
        lr=lr,
        wd=wd,
        momentum=momentum,
        scheduler=scheduler,
        epochs=epochs,
        device=device
    )

    best_val_acc, best_epoch = trainer.train(model_file_name=tmp_file_name)
    print(f"Training complete with best validation accuracy: {best_val_acc}, at epoch: {best_epoch}")

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.load_state_dict(torch.load(tmp_file_name))

    inference(test_dataloader, model, device, result_path=f'{dir_name}/test_predictions.txt')
    print("Inference on test set complete. Predictions saved.")

if __name__ == '__main__':
    app.run(main)
