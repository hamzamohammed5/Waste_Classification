import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf

def load_hists(hists_paths:list):
    """Load model history saved in numpy format"""
    hists = [np.load(i,allow_pickle=True).item() for i in hists_paths]
    histories = []
    for hist in hists:
      try:
        histories.append(hist.history)
      except:
        histories.append(hist)
    return histories

def train_models(models, num_models, t_train_ds, t_val_ds, t_train_steps, t_val_steps, callbacks:list, epochs = 100):
  """Helper function to train several models and return their training histories"""
  # dir to save hist.
  saving_path = '/content/drive/MyDrive/hists'
  if not os.path.exists(saving_path):
    os.makedirs(saving_path)

  hists = []
  # compile and train
  for i, model in enumerate(models):
    print(f"Start training for model: {num_models[i]}")
    model = model()
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy'],
                    run_eagerly=True)
    hist = model.fit(t_train_ds,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=t_val_ds,
                steps_per_epoch=t_train_steps,
                validation_steps=t_val_steps)
    # save the history
    np.save(f'{saving_path}/model-{num_models[i]}-hist.npy', hist.history)
    hists.append(hist.history)
  return hists

def plot_models_metrics(histories=None, num_models=[0,1,2,3], string='loss', skip_epochs = None, figsize = (5,5)):
  """Function to plot accuracy and loss for each model"""
  if not histories:
    # loading histories
    saving_path = '/content/drive/MyDrive/hists'
    histories = load_hists([f'{saving_path}/model-{m}-hist.npy' for m in num_models])

  # plotting histories
  fig1 = plt.figure(figsize=figsize)
  for i,history in enumerate(histories):
    plt.plot(history[string][skip_epochs:])
  plt.ylabel(string)
  plt.legend([f'model {i}: ' for i in num_models])
  plt.xlabel("Epochs")
  plt.title(string)
  fig2 = plt.figure(figsize=figsize)
  for i,history in enumerate(histories):
    plt.plot(history['val_'+string][skip_epochs:])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([f'model {i}: ' for i in num_models])
  plt.title('val_' +string)

def tune_learning_rate(Model, num_model, lrs, t_train_ds, t_val_ds, t_train_steps, t_val_steps, callbacks:list, epochs = 100):
  """Function to tune learning rate for one model """
  saving_path = '/content/drive/MyDrive/lr_hists'
  if not os.path.exists(saving_path):
    os.makedirs(saving_path)
  hists = []
  for lr in lrs:
    print(f'start learning for lr = {lr}')
    model = Model()   # the model creation function
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'],
                  run_eagerly=True)

    hist = model.fit(t_train_ds,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=t_val_ds,
              steps_per_epoch=t_train_steps,
              validation_steps=t_val_steps)
    np.save(f'{saving_path}/model-{num_model}-{lr}-hist.npy', hist.history)
    hists.append(hist.history)
    del(model)
  return hists


def plot_lr_tuning_metric(histories=None, num_model=0, lrs=[0.1,0.001],  string='loss', skip_epochs=None,figsize = (5,5)):
  """Function to plot accuracy and loss for one model """
  if not histories:
    # loading histories
    saving_path = '/content/drive/MyDrive/lr_hists'
    histories = load_hists([f'{saving_path}/model-{num_model}-{lr}-hist.npy' for lr in lrs])

  fig1 = plt.figure(figsize=figsize)
  for history in histories:
    plt.plot(history[string][skip_epochs:])
  plt.ylabel(string)
  plt.legend([f'lr: {i}' for i in lrs])
  plt.xlabel("Epochs")
  plt.title(string)
  fig2 = plt.figure(figsize=figsize)
  for history in histories:
    plt.plot(history['val_'+string][skip_epochs:])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([f'lr: {i}' for i in lrs])
  plt.title('val_' +string)

def plot_model_metic(history = None, string='loss',skip_epochs=None):
  """Helper function to plot model metric for train an val in the same graph"""
  if not history:
    history = pd.read_csv('/content/drive/MyDrive/my_logs.csv')
  plt.plot(history[string])
  plt.plot(history['val_'+string][skip_epochs:])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string][skip_epochs:])
  plt.show()