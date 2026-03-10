import torch
import tensorflow as tf
print(torch.cuda.is_available())
print("GPUs:", tf.config.list_physical_devices('GPU'))