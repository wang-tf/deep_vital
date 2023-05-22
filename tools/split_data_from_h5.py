import os
import h5py
import numpy as np
import argparse
from sklearn.model_selection import train_test_split


def split_index(data_file, save_dir, train_num: int, val_num: int, test_num: int, divide_by_subject: bool, sig_name='ppg'):
    assert os.path.exists(data_file), data_file
    train_num = int(train_num)
    val_num = int(val_num)
    test_num = int(test_num)
    
    with h5py.File(data_file, 'r') as f:
        signal = np.array(f.get(f'/{sig_name}'))
        BP = np.array(f.get('/label'))
        # BP = np.round(BP)
        # BP = np.transpose(BP)
        subject_idx = np.squeeze(np.array(f.get('/subject_idx')))
    N_samp_total = BP.shape[0]
    subject_idx = subject_idx[:N_samp_total]
    print(f'load data samples {N_samp_total}')
    
    # Divide the dataset into training, validation and test set
    # -------------------------------------------------------------------------------
    if divide_by_subject is True:
        valid_idx = np.arange(subject_idx.shape[-1])

        # divide the subjects into training, validation and test subjects
        subject_labels = np.unique(subject_idx)
        subjects_train_labels, subjects_val_labels = train_test_split(subject_labels, test_size=0.5)
        subjects_val_labels, subjects_test_labels = train_test_split(subjects_val_labels, test_size=0.5)

        # Calculate samples belong to training, validation and test subjects
        train_part = valid_idx[np.isin(subject_idx,subjects_train_labels)]
        val_part = valid_idx[np.isin(subject_idx,subjects_val_labels)]
        test_part = valid_idx[np.isin(subject_idx, subjects_test_labels)]

        # draw a number samples defined by N_train, N_val and N_test from the training, validation and test subjects
        idx_train = np.random.choice(train_part, train_num, replace=False)
        idx_val = np.random.choice(val_part, val_num, replace=False)
        idx_test = np.random.choice(test_part, test_num, replace=False)
    else:
        # Create a subset of the whole dataset by drawing a number of subjects from the dataset. The total number of
        # samples contributed by those subjects must equal N_train + N_val + _N_test
        subject_labels, SampSubject_hist = np.unique(subject_idx, return_counts=True)
        cumsum_samp = np.cumsum(SampSubject_hist)
        subject_labels_train = subject_labels[:np.nonzero(cumsum_samp>(train_num+val_num+test_num))[0][0]]
        idx_valid = np.nonzero(np.isin(subject_idx,subject_labels_train))[0]

        # divide subset randomly into training, validation and test set
        idx_train, idx_val = train_test_split(idx_valid, train_size= train_num, test_size=val_num+test_num)
        idx_val, idx_test = train_test_split(idx_val, test_size=0.5)
    print(f'train data num: {len(idx_train)}')
    print(f'val data num: {len(idx_val)}')
    print(f'test data num: {len(idx_test)}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_index_save_path = os.path.join(save_dir, 'train_index.txt')
    val_index_save_path = os.path.join(save_dir, 'val_index.txt')
    test_index_save_path = os.path.join(save_dir, 'test_index.txt')
    np.savetxt(train_index_save_path, idx_train, delimiter=',', fmt='%d')
    np.savetxt(val_index_save_path, idx_val, delimiter=',', fmt='%d')
    np.savetxt(test_index_save_path, idx_test, delimiter=',', fmt='%d')
    
    train_data_save_path = os.path.join(save_dir, 'train.h5')
    val_data_save_path = os.path.join(save_dir, 'val.h5')
    test_data_save_path = os.path.join(save_dir, 'test.h5')
    with h5py.File(train_data_save_path, 'w') as f:
        f.create_dataset(sig_name, data=signal[idx_train])
        f.create_dataset('label', data=BP[idx_train])
        f.create_dataset('subject_idx', data=subject_idx[idx_train])
    with h5py.File(val_data_save_path, 'w') as f:
        f.create_dataset(sig_name, data=signal[idx_val])
        f.create_dataset('label', data=BP[idx_val])
        f.create_dataset('subject_idx', data=subject_idx[idx_val])
    with h5py.File(test_data_save_path, 'w') as f:
        f.create_dataset(sig_name, data=signal[idx_test])
        f.create_dataset('label', data=BP[idx_test])
        f.create_dataset('subject_idx', data=subject_idx[idx_test])
        
    return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('save_dir')
    parser.add_argument('--train_num', default=1e6)
    parser.add_argument('--val_num', default=2.5e5)
    parser.add_argument('--test_num', default=2.5e5)
    parser.add_argument('--divide', default=True)
    parser.add_argument('--sig_name', default='ppg')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(args)
    data_file = args.data_file
    save_dir = args.save_dir
    train_num = args.train_num
    val_num = args.val_num
    test_num = args.test_num
    divide_by_subject=args.divide
    sig_name = args.sig_name
    split_index(data_file, save_dir, train_num, val_num, test_num, divide_by_subject, sig_name)


if __name__ == '__main__':
    main()
