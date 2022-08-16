import general 
import h5py 
import s3fs 
import csv 
import numpy as np 
import os 
import fsspec 
from fsspec import utils 

def get_file_names(file_paths):
    return [os.path.basename(path) for path in file_paths]

def get_file_paths(file_names, dir_path):
    return [os.path.join(dir_path, name) for name in file_names]

def copy_batch(fs, batch, src_dir, dest_dir, dry_run):
    # copies files in the batch from src to dest, 
    # deletes files from src if copy was successful 

    batch_src_paths = get_file_paths(batch, src_dir)
    batch_dest_paths = get_file_paths(batch, dest_dir)
    #print(f"Copying batch with files: {batch}")
    paths = fs.expand_path(batch_src_paths)
    path2 = utils.other_paths(paths, batch_dest_paths)
    paths_files = np.array([x.split("/")[-1] for x in paths])
    path2_files = np.array([x.split("/")[-1] for x in path2])
    eq_test = paths_files == path2_files
    print("Equal paths", eq_test.all())
    
    fs.cp(batch_src_paths, batch_dest_paths)
    
    dest_dir_files = get_file_names(fs.ls(dest_dir))

    # finds files that were specified in the batch but not in dest dir
    not_copied_over = set(batch).difference(dest_dir_files)
    if len(not_copied_over) > 0:
        print(f"WARNING: {len(not_copied_over)} files not copied over {not_copied_over}")

    # finds files that were specified in the batch and also in dest dir 
    # (successfully copied)
    copied_over = set(batch).intersection(dest_dir_files)
    # can delete those
    if len(copied_over) > 0:
        copied_over_paths = get_file_paths(copied_over, src_dir)
        #print(f"Deleting {len(copied_over_paths)} files that have been copied over: {copied_over_paths}")
        fs.rm(copied_over_paths)

if __name__ == "__main__":
    subject = "SA"
    session = "20180802"
    
    fs = s3fs.S3FileSystem()
    for x in fs.glob("l2l.mfliu.scratch/dest"):
        fs.rm(x)

    for x in range(1, 1009):
        with fs.open("l2l.mfliu.scratch/source/test" + str(x) + ".txt", 'wb') as f:
            np.save(f, x * np.ones(shape=(1000,)))

    batch_size = 10
    all_files = [x.split("source/")[-1] for x in fs.glob("l2l.mfliu.scratch/source/*.txt")]
    batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]
    src_dir = "l2l.mfliu.scratch/source/"
    dest_dir = "l2l.mfliu.scratch/dest/"
    for batch in batches:
        copy_batch(fs, batch, src_dir, dest_dir, False)
    
    #for x in fs.glob("l2l.mfliu.scratch/dest/*.txt"):
    #    print(x, np.load(fs.open(x)))

    """
    all_unit_info = general.list_session_units(fs, subject, session)
    with open('erroneous_sptime.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        csvwriter.writerow(["FileName", "FileKeys", "WFShape", "CorrespondingWaveformFile",
            "WaveformFileKeys"])
        for test_file in all_unit_info.SpikeTimesFile:
            test_data = h5py.File(fs.open(test_file))
            if 'timestamps' not in test_data.keys():
                test_file_wf = test_file.split("_spiketimes.mat")[0] + "_waveforms.mat"
                test_data_wf = h5py.File(fs.open(test_file_wf))

                print(test_file, test_data.keys(), test_data['waveforms'].shape, test_file_wf,\
                        test_data_wf.keys())
                csvwriter.writerow([test_file, test_data.keys(), test_data['waveforms'].shape,\
                        test_file_wf, test_data_wf.keys()])

    """
