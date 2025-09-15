import os
import audb
import audiofile
import opensmile

db = audb.load(
    "emodb",
    version="1.1.1",
    format="wav",
    mixdown=True,
    sampling_rate=16000,
    media="wav/03a01.*",
    full_path=False,
    verbose=False
)

for dirpath, dirnames, filenames in os.walk("./dataset/ADReSS-IS2020-data/train/full_preprocessing/cd/processed_chunks"):
    folder_name = os.path.basename(dirpath)
    log_dir = os.path.join("./SALSA/benchmark/processed_features/train/cd", folder_name, "log")
    feat_dir = os.path.join("./SALSA/benchmark/processed_features/train/cd", folder_name, "features")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    for filename in filenames:
        if filename.startswith("segment_") and filename.endswith(".wav"):
            input_file_path = os.path.join(dirpath, filename)
            log_file_path = os.path.join(log_dir, filename.replace(".wav", ".log"))
            feat_file_path = os.path.join(feat_dir, filename.replace(".wav", ".txt"))

            signal, sampling_rate = audiofile.read(input_file_path, duration=10, always_2d=True)
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
                loglevel=2,
                logfile=log_file_path
            )

            features = smile.process_signal(signal, sampling_rate)
            with open(feat_file_path, "w") as fp:
                fp.write(str(features))