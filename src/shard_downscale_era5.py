import glob
import os

import click
import numpy as np
from tqdm import tqdm


def sort_one_year(file_list):
    shard_ids = [int(f.split(".")[0].split("_")[1]) for f in file_list]
    sorted_ids = np.argsort(shard_ids)
    return np.array(file_list)[sorted_ids]


def sort_shards(root_dir, orig_n_shards):
    ps = glob.glob(os.path.join(root_dir, f"*.npz"))
    all_files = [os.path.basename(p) for p in ps]

    files_each_year = []
    for i in range(0, len(all_files), orig_n_shards):
        files_each_year.append(all_files[i : i + orig_n_shards])

    files_each_year = [sort_one_year(year_list) for year_list in files_each_year]

    all_files = np.concatenate(files_each_year)
    all_files = [os.path.join(root_dir, f) for f in all_files]

    return all_files


def shard_downscale(
    root_dir, highres_root_dir, save_dir, orig_n_shards, highres_orig_n_shards, num_per_shard, partition="train"
):
    """
    root_dir: path to directory that stores npz files of the low-res data
    highres_root_dir: path to directory that stores npz files of the high-res data
    pred_range: forecast range in hours
    orig_n_shards: the number of shards for each year in root_dir
    highres_orig_n_shards: the number of shards for each year in root_dir highres_root_dir
    num_per_shard: number of (inp, out) pairs per shard
    """
    if not os.path.exists(os.path.join(save_dir, partition)):
        os.makedirs(os.path.join(save_dir, partition))

    # sort npy files in chronological order
    lowres_files = sort_shards(os.path.join(root_dir, partition), orig_n_shards)
    highres_files = sort_shards(os.path.join(highres_root_dir, partition), highres_orig_n_shards)

    sample = np.load(lowres_files[0])
    all_vars = list(sample.keys())

    # buffer to store end-of-file data points
    data_buffer = {var: [] for var in all_vars}
    highres_data_buffer = {var: [] for var in all_vars}
    shard_id = 0

    num_files = max(len(lowres_files), len(highres_files))

    for i in tqdm(range(num_files)):

        if i < len(lowres_files):
            npy_data = np.load(lowres_files[i])
            # load data from npz file and add to data buffer
            for k in npy_data.keys():
                data_buffer[k].extend(npy_data[k])

        if i < len(highres_files):
            highres_npy_data = np.load(highres_files[i])

            # load data from npz file and add to data buffer
            for k in highres_npy_data.keys():
                highres_data_buffer[k].extend(highres_npy_data[k])

        # if data buffer has enough data points to create a shard
        while len(data_buffer[all_vars[0]]) >= num_per_shard and len(highres_data_buffer[all_vars[0]]) >= num_per_shard:
            print("len buffer", len(data_buffer[all_vars[0]]))
            print("len highres buffer", len(highres_data_buffer[all_vars[0]]))
            shard_inp = {k: np.stack(v[:num_per_shard], axis=0) for k, v in data_buffer.items()}
            shard_out = {k: np.stack(v[:num_per_shard], axis=0) for k, v in highres_data_buffer.items()}
            print("shard input shape", shard_inp[all_vars[0]].shape)
            print("shard output shape", shard_out[all_vars[0]].shape)
            data_buffer = {k: v[num_per_shard:] for k, v in data_buffer.items()}
            highres_data_buffer = {k: v[num_per_shard:] for k, v in highres_data_buffer.items()}
            print("len buffer", len(data_buffer[all_vars[0]]))
            print("len highres buffer", len(highres_data_buffer[all_vars[0]]))
            print("=" * 100)

            # save the created shard
            shard_name = str(shard_id).zfill(3)
            np.savez(os.path.join(save_dir, partition, f"{shard_name}_inp.npz"), **shard_inp)
            np.savez(os.path.join(save_dir, partition, f"{shard_name}_out.npz"), **shard_out)

            shard_id += 1


@click.command()
@click.argument("root_dir", type=click.Path(exists=True))
@click.argument("highres_root_dir", type=click.Path(exists=True))
@click.option("--save_dir", type=str)
@click.option("--orig_n_shards", type=int, default=12)
@click.option("--highres_orig_n_shards", type=int, default=32)
@click.option("--val_n_shards", type=int, default=32)
def main(
    root_dir,
    highres_root_dir,
    save_dir,
    orig_n_shards,
    highres_orig_n_shards,
    val_n_shards,
):
    # compute # data points per shard
    all_files = glob.glob(os.path.join(root_dir, "val", f"*.npz"))
    sample = np.load(all_files[0])
    n_points = sample[list(sample.keys())[0]].shape[0] * len(all_files)
    num_per_shard = n_points // val_n_shards

    shard_downscale(root_dir, highres_root_dir, save_dir, orig_n_shards, highres_orig_n_shards, num_per_shard, "train")
    shard_downscale(root_dir, highres_root_dir, save_dir, orig_n_shards, highres_orig_n_shards, num_per_shard, "val")
    shard_downscale(root_dir, highres_root_dir, save_dir, orig_n_shards, highres_orig_n_shards, num_per_shard, "test")


if __name__ == "__main__":
    main()
