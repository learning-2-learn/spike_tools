import general 
import h5py 
import s3fs 
import csv 

if __name__ == "__main__":
    subject = "SA"
    session = "20180802"
    
    fs = s3fs.S3FileSystem()
    
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

