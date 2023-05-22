import argparse
from pathlib import Path

import torch
import yaml
from easydict import EasyDict

from qartezator.trainers import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, help="Path to config yml", required=True)
    parser.add_argument("--ckpt-path", type=str, help="Path to checkpoint", required=True)
    parser.add_argument("--output-path", type=Path, help="Output path", required=True)
    return parser.parse_args()


def main():
    args = get_args()

    output_path = args.output_path
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(args.config_path, 'r') as f:
        try:
            config = EasyDict(yaml.safe_load(f))
        except yaml.YAMLError as exc:
            print(exc)

    config.training_model.predict_only = True
    config.visualizer.kind = 'noop'

    model = load_checkpoint(config, args.ckpt_path, strict=False, map_location="cpu")
    model.eval()

    image = torch.randn(1, 3, 256, 256)

    if not torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    image = image.to(device)
    model = model.to(device)

    traced_model = torch.jit.trace(model.generator, image, strict=False).to(device)
    torch.jit.save(traced_model, output_path)


if __name__ == "__main__":
    main()
