import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# # Preview the result
# for val in dataset:
#    print(val.numpy())

# # Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
# dataset = tf.data.Dataset.range(10)

# # Window the data
# dataset = dataset.window(size=5, shift=1)

# # Print the result
# for window_dataset in dataset:
#   print(window_dataset)

# # Print the result
# for window_dataset in dataset:
#   print([item.numpy() for item in window_dataset])


# # Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
# dataset = tf.data.Dataset.range(10)

# # Window the data but only take those with the specified size
# dataset = dataset.window(size=5, shift=1, drop_remainder=True)

# # Print the result
# for window_dataset in dataset:
#   print([item.numpy() for item in window_dataset])

# # Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
# dataset = tf.data.Dataset.range(10)

# # Window the data but only take those with the specified size
# dataset = dataset.window(5, shift=1, drop_remainder=True)

# # Flatten the windows by putting its elements in a single batch
# dataset = dataset.flat_map(lambda window: window.batch(5))

# # Print the results
# for window in dataset:
#   print(window.numpy())



# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
dataset = dataset.window(5, shift=1, drop_remainder=True)

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

# Create tuples with features (first four elements of the window) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# Shuffle the windows
dataset = dataset.shuffle(buffer_size=10)

# Create batches of windows
dataset = dataset.batch(2).prefetch(1)

# Print the results
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
  print()



"""
## Wrap Up

This short exercise showed you how to chain different methods of the `tf.data.Dataset` 
class to prepare a sequence into shuffled and batched window datasets. 
You will be using this same concept in the next exercises when you apply 
it to synthetic data and use the result to train a neural network. On to the next!

"""