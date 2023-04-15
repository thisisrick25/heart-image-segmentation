import splitfolders

# Split with a ratio of 70/30
splitfolders.ratio("datasets/Task02_Heart/nifti_files/",
                   output="datasets/Task02_Heart/data_train_test",
                   seed=1337, move=False, ratio=(.7, .3))
