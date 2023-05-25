import argparse
import glob
import shutil
from pathlib import Path

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist-path", type=Path, help="Path to filelist", required=True)
    parser.add_argument("--source-dir", type=Path, help="Source dir path", required=True)
    parser.add_argument("--dest-dir", type=Path, help="Dest dir path", required=True)
    parser.add_argument("--ignore-ext", action='store_true', default=False)
    parser.add_argument("--max-items", type=int, help="Num of samples.", default=None, required=False)
    return parser.parse_args()


def main():
    args = get_args()
    dest_dir = args.dest_dir
    dest_dir.mkdir(exist_ok=True, parents=True)

    with open(args.filelist_path, "r") as f:
        names = f.read().splitlines()

    for name in tqdm(names):
        try:
            if args.ignore_ext:
                source_file_paths = glob.glob(f"{args.source_dir / Path(name).stem}*")
                for source_file_path in source_file_paths:
                    shutil.copy(source_file_path, dest_dir)
            else:
                source_file_path = args.source_dir / Path(name).name
                shutil.copy(source_file_path, dest_dir)
        except (FileNotFoundError, IndexError):
            print(f"{Path(name).name} file was not found in dir {args.source_dir}")


if __name__ == "__main__":
    main()
