import torch
import tensorflow as tf
import numpy as np
from natsort import natsorted
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)

"""
subset input data X (embeddings) to 100 entries from validation
"""
# path_inputdata = '/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_newmodel/embeddings_validation_pretrainedmodel.pt'
# tensor_inputdata = torch.load(path_inputdata, map_location=torch.device(device))
# print(f'shape of tensor with input data: {tensor_inputdata.shape}')
# # subset first 100 entries
# tensor_inputdata_small = tensor_inputdata[:100]
# print(f'shape of tensor with input data small: {tensor_inputdata_small.shape}')
# # torch.save(tensor_inputdata_small, 'tensor_embeddingsvalidation_100.pt')
# print(torch.equal(tensor_inputdata[0], tensor_inputdata_small[0]))

### show first tensor of input data X
t_input = torch.load('tensor_embeddingsvalidation_100.pt', map_location=torch.device(device))
print(f'first tensor of embeddings input X')
print(f'shape: {t_input.shape}')
print(f'first tensor:')
print(f'{t_input[0]}')

"""
load tfr records (target) from validation
"""
print(f'\nloading tensor flow records (targets Y)')
# select tensorflow records for validation sequences
subset = 'valid'
tfr_path = f'/exports/humgen/idenhond/data/Basenji/tfrecords/{subset}*.tfr'
tfr_files = natsorted(glob.glob(tfr_path))
print(tfr_files)
dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
print(dataset)

# metadata human --> invullen in code
{
    "num_targets": 5313,
    "train_seqs": 34021,
    "valid_seqs": 2213,
    "test_seqs": 1937,
    "seq_length": 131072,
    "pool_width": 128,
    "crop_bp": 8192,
    "target_length": 896
}

# function to get info out of the records
def make_parser(): #, rna_mode
    def parse_proto(example_protos):
        """Parse TFRecord protobuf."""

        feature_spec = {
        'sequence': tf.io.FixedLenFeature([], dtype = tf.string),  # Ignore this, resize our own bigger one
        'target': tf.io.FixedLenFeature([], dtype = tf.string),
        }

        # parse example into features
        feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)

        sequence = tf.io.decode_raw(feature_tensors['sequence'], tf.bool)
        sequence = tf.reshape(sequence, (131072, 4))
        sequence = tf.cast(sequence, tf.float32)

        target = tf.io.decode_raw(feature_tensors['target'], tf.float16)
        target = tf.reshape(target, (896, 5313))
        target = tf.cast(target, tf.float32)

        return sequence, target

    return parse_proto

def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

dataset = dataset.flat_map(file_to_records)
dataset = dataset.map(make_parser())
dataset = dataset.batch(1)
# print(dataset)
# print(type(dataset))

# initialize inputs and outputs
seqs_1hot = []
targets = []

# collect inputs and outputs
teller = 0
for seq_1hot, targets1 in dataset:
    # dataset for test sequences contains 1937 sets of sequences + targets
    # sequence
    print(teller)
    teller += 1
    # seqs_1hot.append(seq_1hot.numpy())
    print(type(targets1.numpy()))
    targets.append(targets1.numpy())
    # print(f'shape sequence: {seq_1hot.numpy().shape}')
    # print(f'shape target: {targets1.numpy().shape}')
    if teller == 100:
        break
    

# make arrays
# seqs_1hot = np.array(seqs_1hot)
targets = np.array(targets)
targets = np.squeeze(targets)

# print(seqs_1hot[0])

# print(f'shape sequence: {seqs_1hot.shape}')
print(f'shape target: {targets.shape}')
# print(targets[0])

tensor_target_validation_100 = torch.from_numpy(targets)

print(f'\nfirst tensor of target output Y')
print(f'shape: {tensor_target_validation_100.shape}')
print(f'first tensor:')
print(f'{tensor_target_validation_100[0]}')

print('\n')
print(tensor_target_validation_100.shape)
print(type(tensor_target_validation_100))
torch.save(tensor_target_validation_100, ' .pt')

# show output validation pretrained model
# path = '/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_output_newmodel/output_validation.pt'
# t = torch.load(path, map_location=torch.device('cpu'))
# print(f'shape of output validation tensor: {t.shape}')
# print(t[0])
