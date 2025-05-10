import tensorflow as tf
#from tensorflow.python.ops import spectral_ops

import os
import shutil

def recreate_directory(dir_path):
    # Si el directorio existe, lo elimina
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    # Crea el directorio de nuevo
    os.makedirs(dir_path)
    print(f"Directorio '{dir_path}' creado (si existía, fue borrado primero).")



def conv1d(x, nfilter, filter_size, step):
    return tf.layers.conv1d(inputs=x, filters=nfilter, kernel_size=filter_size, strides=step, padding='SAME')

def conv2d(x, nfilter, filter_size, step):
    return tf.layers.conv2d(inputs=x, filters=nfilter, kernel_size=filter_size, strides=step, padding='SAME')

def conv2d_transpose(x, filter_size, out_shape, step, name='deconv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', filter_size, initializer=tf.random_normal_initializer(stddev=0.01))
    return tf.nn.conv2d_transpose(value=x, filter=w, output_shape=out_shape, strides=step)

def ResBlock(x, nfilters, filter_size, step):
    output = x
    hidden = tf.layers.conv1d(inputs=x, filters=nfilters, kernel_size=filter_size, strides=step, padding='SAME')
    return tf.add(output, hidden)

def compose(x, nfilter, kernel_size, out_size):
    temp = tf.layers.conv1d(inputs=x, filters=nfilter, kernel_size=kernel_size, strides=1, padding='SAME')
    temp = tf.nn.leaky_relu(temp)

    in_size = x.shape[1].value
    avg_kernel_size = in_size * nfilter // out_size
    temp = tf.transpose(temp, perm=[0, 2, 1])
    temp = avgpool(temp, avg_kernel_size, avg_kernel_size)

    output = tf.reshape(temp, shape=[-1, out_size, 1])
    return output

def avgpool1d(x, pool_size, step):
    return tf.layers.average_pooling1d(inputs=x, pool_size=pool_size, strides=step, padding='SAME')

def maxpool1d(x, pool_size, step):
    return tf.layers.max_pooling1d(inputs=x, pool_size=pool_size, strides=step, padding='SAME')

def maxpool2d(x, pool_size, step):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=step, padding='SAME')

def get_median(v):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0] // 2
    return tf.nn.top_k(v, m).values[m-1]

def tensor_stft(x, FrmLeng, FrmOver):
    FrmStep = FrmLeng-FrmOver
    tensor_spec = tf.contrib.signal.stft(signals=x, frame_length=FrmLeng, frame_step=FrmStep,
                                         fft_length=FrmLeng, window_fn=tf.contrib.signal.hamming_window)
    tensor_spec = tf.abs(tensor_spec)
    tensor_spec = tf.transpose(a=tensor_spec, perm=[0, 2, 1])
    nFre, nFrm = tensor_spec.shape[1], tensor_spec.shape[2]
    return tf.reshape(tensor=tensor_spec, shape=[-1, nFre, nFrm, 1])

def tensor_fft(x, FrmLeng):
    nData, nFrm = x.shape[0], x.shape[1]
    nFre = int(FrmLeng/2)+1
    tensor_spectrogram = []
    for dter in range(nData):
        tensor_spectrum = []
        for fter in range(nFrm):
            temp_spec = spectral_ops.rfft(x[dter, fter, :FrmLeng], [FrmLeng])
            temp_spec = tf.abs(temp_spec)
            tensor_spectrum.append(temp_spec)
        tensor_spectrum = tf.convert_to_tensor(tensor_spectrum, dtype=tf.float32)
        tensor_spectrogram.append(tensor_spectrum)
    tensor_spectrogram = tf.convert_to_tensor(tensor_spectrogram)
    return tf.reshape(tensor=tensor_spectrogram, shape=[-1, nFrm, nFre, 1])

def tensor_normalize(input_):
    nData, nDim = input_.shape[0], input_.shape[1]

    maxVal = tf.reduce_max(input_tensor=tf.abs(input_), axis=1)
    maxVal = tf.reshape(tensor=maxVal, shape=[nData, 1])
    maxVal = tf.matmul(maxVal, tf.ones(shape=[1, nDim]))

    output = tf.div(input_, maxVal)

    return output


import os, re
import numpy as np
import scipy.signal as sp
import wave, struct
from scipy.io import wavfile
import matplotlib.pyplot as plt

def WaveRead(dirPath):
    wave_list = []
    for (path, dir, files) in os.walk(dirPath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.wav':
                wave_list.append(path + '/' + filename)
    wave_list.sort()

    nData = len(wave_list)
    output = []
    for dter in range(nData):
        fs, wavData = wavfile.read(wave_list[dter])
        output.append(wavData)

    output = np.asarray(output)
    nLength = output.shape[1]
    return output, nData, nLength

def WaveNormalization(target):
    nData, nDim = target.shape[0], target.shape[1]

    tempMean = np.mean(target, axis=1)
    tempStd  = np.std(target, axis=1, ddof=1)

    out = np.zeros(shape=[nData, nDim], dtype=float)
    for dter in range(nData):
        out[dter,:nDim] = (target[dter,:nDim] - tempMean[dter]) / tempStd[dter]

    return out

def SpecNormToImage(target):
    nData, nDim, nFrm = target.shape[0], target.shape[1], target.shape[2]

    output = []
    for dter in range(nData):
        tempMax = np.max(np.max(target[dter], axis=1))
        tempMin = np.min(np.min(target[dter], axis=1))

        temp = (target[dter] - tempMin) / (tempMax - tempMin)
        output.append(temp)

    output = np.asarray(output)

    nFre, nFrm = output.shape[1], output.shape[2]
    return np.reshape(output, [-1, nFre, nFrm, 1]), nFre, nFrm

def sequence2frame(target, frame_size, frame_over):
    nData = target.shape[0]
    target = np.concatenate([target, np.zeros(shape=[nData, 100])], axis=1)
    nDim = target.shape[1]
    frame_shift = frame_size - frame_over
    nFrame = int((nDim-frame_size)/frame_shift)

    out = np.zeros(shape=[nData, frame_size, nFrame])
    for dter in range(nData):
        for fter in range(nFrame):
            stridx = fter * frame_shift
            endidx = stridx + (frame_size-1)
            out[dter, :(frame_size-1), fter] = target[dter, stridx:endidx]
    return out, nFrame

def frame2sequence(frames, frame_size, frame_over):
    hwindow = np.hamming(frame_size)
    nData, nFrm = frames.shape[0], frames.shape[1]
    frame_shift = frame_size - frame_over
    nLength = (nFrm-1)*frame_shift + frame_size
    output = np.zeros(shape=[nData, nLength], dtype=float)
    for dter in range(nData):
        for fter in range(nFrm):
            stridx = fter*frame_shift
            endidx = stridx + frame_size
            temp_frame = np.multiply(frames[dter, fter, :frame_size], hwindow)
            output[dter, stridx:endidx] = output[dter, stridx:endidx] + temp_frame

    return output

def WriteWave(savePath, nchannels, sampwidth, FS, value, maxValue):
    wav_fp = wave.open(savePath, 'w')
    wav_fp.setnchannels(nchannels)
    wav_fp.setsampwidth(sampwidth)
    wav_fp.setframerate(FS)
    for j in range(value.size):
        sample = int(maxValue * value[j])
        data = struct.pack('<h', sample)
        wav_fp.writeframesraw(data)
    wav_fp.close()


def wav2spec(wavedata, FS, frmLeng, frmOver):
    nData, nLength = wavedata.shape[0], wavedata.shape[1]

    win = sp.get_window('hamming', frmLeng)
    specdata = []
    for dter in range(nData):
        f, t, tempspec = sp.spectrogram(x=wavedata[dter], fs=FS, window=win, nperseg=frmLeng, noverlap=frmOver)
        specdata.append(tempspec)
    specdata = np.asarray(specdata)

    nFre, nFrm = specdata.shape[1], specdata.shape[2]
    return np.reshape(specdata, [-1, nFre, nFrm, 1]), nFre, nFrm

# import packages
import argparse
#from ops import *
#from utilities import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--working_dir', type=str, default='.')
    parser.add_argument('--model_name', type=str, default='cnn_wgan')
    parser.add_argument('--batch_size', type=int, default=32)   # ICML ~ 16
    parser.add_argument('--max_epoch', type=int, default=5000)
    parser.add_argument('--eps_stop', type=float, default=0.01)
    parser.add_argument('--mode', type=int, default=1) # 1 for training, 2 generation
    parser.add_argument('--batch_size_gen', type=int, default=10) # 1 for training, 2 generation
    args = parser.parse_args()


working_dir = args.working_dir
mode = args.mode
model_name = args.model_name
eps_stop = args.eps_stop
batch_size_gen = args.batch_size_gen
flagFirstEpoch = True

if mode == 1:
    # create dirs to update results
    recreate_directory(working_dir+"/model/WGAN")
    recreate_directory(working_dir+"/result")
    recreate_directory(working_dir+"/wave_log")
    recreate_directory(working_dir+"/wave/WGAN")

# real data loading
wavDir = working_dir+"/audio"
WavData, nData, nLength = WaveRead(wavDir)
WavData = WaveNormalization(WavData)

print("---------->>>>Audio length:"+str(WavData.shape[1]))

# audio parameters
FS = 16000
FrmLeng = 512
FrmOver = int(FrmLeng * 3 / 4)
total_epochs = args.max_epoch
maxValue = 32767                              # max value of short integer(2 byte)

# transform from wave to spectrogram
SpecData, nFre, nFrm = wav2spec(WavData, FS, FrmLeng, FrmOver)

# training parameters
batch_size = args.batch_size
learning_rate = 0.000001

# generating parameters
random_dim = 128

# module 1: Generator
def generator(z):
    with tf.variable_scope(name_or_scope="G") as scope:
        # define weights for generator
        weights = {
            'gw1': tf.get_variable(name='gw1', shape=[random_dim, FrmLeng], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'gw2': tf.get_variable(name='gw2', shape=[FrmLeng, FrmLeng], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'gw3': tf.get_variable(name='gw3', shape=[FrmLeng, int(FrmLeng)], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        }
        bias = {
            'gb1': tf.get_variable(name='gb1', shape=[FrmLeng], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'gb2': tf.get_variable(name='gb2', shape=[FrmLeng], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),
            'gb3': tf.get_variable(name='gb3', shape=[FrmLeng], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        }
    gw4 = tf.get_variable(name='gw4', shape=[int(FrmLeng/2), nLength], dtype=tf.float32,
                          initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

    fc = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(z, weights['gw1']), bias['gb1'])))
    fc = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(fc, weights['gw2']), bias['gb2'])))
    fc = tf.cos(tf.layers.batch_normalization(tf.add(tf.matmul(fc, weights['gw3']), bias['gb3'])))

    fc1 = tf.slice(input_=fc, begin=[0, 0], size=[batch_size, int(FrmLeng/2)])
    fc2 = tf.slice(input_=fc, begin=[0, int(FrmLeng/2)], size=[batch_size, int(FrmLeng/2)])

    fc = tf.add(tf.matmul(fc1, gw4), tf.matmul(fc2, gw4))

    return tf.nn.tanh(fc)

# module 2: Discriminator
def discriminator(x, reuse=False):
    if reuse == False:
        with tf.variable_scope(name_or_scope="D") as scope:
            weights = {
                'dw1': tf.get_variable(name='dw1', shape=[17 * 4 * 16, 1], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            }
            bias = {
                'db1': tf.get_variable(name='db1', shape=[1], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            }
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(x, 2, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 4, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 8, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 16, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
    else:
        with tf.variable_scope(name_or_scope="D", reuse=True) as scope:
            weights = {
                'dw1': tf.get_variable(name='dw1', shape=[17 * 4 * 16, 1], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            }
            bias = {
                'db1': tf.get_variable(name='db1', shape=[1], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            }
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(x, 2, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 4, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 8, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])
            hconv = tf.nn.relu(tf.layers.batch_normalization(conv2d(hconv, 16, [3, 3], [1, 1])))
            hconv = maxpool2d(hconv, [2, 2], [2, 2])

    hconv = tf.reshape(hconv, shape=[-1, 17 * 4 * 16])
    output = tf.nn.sigmoid(tf.add(tf.matmul(hconv, weights['dw1']), bias['db1']))

    return output

# module 3: Random noise as an input
def random_noise(batch_size):
    return np.random.normal(size=[batch_size, random_dim]), np.zeros(shape=[batch_size, 1])

# Make a graph
g = tf.Graph()
with g.as_default():
    # input node
    X = tf.placeholder(tf.float32, [batch_size, nFre, nFrm, 1])       # for real data
    Z = tf.placeholder(tf.float32, [batch_size, random_dim])    # for generated samples

    # Results in each module; G and D
    fake_x = generator(Z)
    fake_spec = tensor_stft(fake_x, FrmLeng=FrmLeng, FrmOver=FrmOver)

    # Probability in discriminator
    result_of_fake = discriminator(fake_spec)
    result_of_real = discriminator(X, True)

    # for WGAN: Loss function in each module: G and D => it must be maximize
    g_loss = tf.reduce_mean(result_of_fake)
    d_loss = tf.reduce_mean(result_of_real) - tf.reduce_mean(result_of_fake)

    # Optimization procedure
    t_vars = tf.trainable_variables()

    gr_vars = [var for var in t_vars if "gw4" in var.name]
    dc_vars = [var for var in t_vars if ("dw1" or "db1") in var.name]

    g_vars = [var for var in t_vars if "G" in var.name]
    d_vars = [var for var in t_vars if "D" in var.name]
    w_vars = [var for var in t_vars if ("D" or "G") in var.name]

    # Regularization for weights
    gr_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l1_regularizer(1.0e-6),
                                                     weights_list=gr_vars)
    g_loss_reg = g_loss - gr_loss
    d_loss_reg = d_loss

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    g_train = optimizer.minimize(-g_loss_reg, var_list=g_vars)
    gw_train = optimizer.minimize(-g_loss_reg, var_list=gr_vars)
    d_train = optimizer.minimize(-d_loss_reg, var_list=d_vars)

    d_clip = [v.assign(tf.clip_by_value(v, -0.005, 0.005)) for v in dc_vars]

# Training graph g
saver = tf.train.Saver(var_list=w_vars)
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(working_dir+'/model/WGAN')
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(working_dir+'/model/WGAN', ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
    else:
        counter = 0

    total_batchs = int(WavData.shape[0] / batch_size)

    logPath = working_dir+"/result/GAN_result.log"
    log_fp = open(logPath, 'w')
    log = "Class: %s, nData: %d, max_epoch: %d, batch_size: %d, random_dim: %d" \
          % (model_name, nData, total_epochs, batch_size, random_dim)
    print(log)
    log_fp.write(log + "\n")

    for epoch in range(counter, total_epochs):
        avg_G_loss = 0
        avg_D_loss = 0

        data_indices = np.arange(nData)
        np.random.shuffle(data_indices)
        SpecData = SpecData[data_indices]
        for batch in range(total_batchs):
            batch_x = SpecData[batch*batch_size:(batch+1)*batch_size]

            noise, nlabel = random_noise(batch_size)
            sess.run(d_train, feed_dict={X: batch_x, Z: noise})
            sess.run(d_clip)

            sess.run(g_train, feed_dict={Z: noise})
            sess.run(gw_train, feed_dict={Z: noise})
            sess.run(gw_train, feed_dict={Z: noise})

            gl, dl = sess.run([g_loss_reg, d_loss_reg], feed_dict={X: batch_x, Z: noise})

            avg_G_loss += gl
            avg_D_loss += dl

        avg_G_loss /= total_batchs
        avg_D_loss /= total_batchs        

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log = "=========Epoch : %d ======================================" % (epoch + 1)
            print(log)
            log_fp.write(log + "\n")
            log = "G_loss : %.15f" % avg_G_loss
            print(log)
            log_fp.write(log + "\n")
            log = "D_loss : %.15f" % avg_D_loss
            print(log)
            log_fp.write(log + "\n")

            # Generating wave
            sample_input, _ = random_noise(batch_size)
            generated = sess.run(fake_x, feed_dict={Z: sample_input})

            # Writing the generated wave
            savePath = working_dir+'/wave_log/{}.wav'.format(str(epoch + 1).zfill(3))
            WriteWave(savePath, 1, 2, FS, generated[0], maxValue)
            log = "Writing generated audio to %s" % savePath
            print(log)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # save model
            modelPath = working_dir+"/model/WGAN/{}".format(model_name)
            saver.save(sess=sess, save_path=modelPath, global_step=(epoch + 1))

        # Check for early stop
        if flagFirstEpoch == True:
            avg_G_loss_old = avg_G_loss
            avg_D_loss_old = avg_D_loss
            flagFirstEpoch = False
        else:
            if abs(avg_G_loss_old - avg_G_loss)<eps_stop:
                print('Stop by early stopping with epsilon of generator:'+str(eps_stop))
                break
            
        avg_G_loss_old = avg_G_loss
        avg_D_loss_old = avg_D_loss



    # Generating wave
    for ibatch in range(batch_size_gen):
        sample_noise, _ = random_noise(batch_size)
        generated = sess.run(fake_x, feed_dict={Z: sample_noise})
        # Writing the generated wave
        for i in range(batch_size):
            savePath = working_dir+'/wave/WGAN/{}_{}.wav'.format(str(i).zfill(3),ibatch)
            WriteWave(savePath, 1, 2, FS, generated[i], maxValue)
            print("Writing generated audio to " + savePath)

    log = "Complete Audio GAN"
    print(log)
    log_fp.write(log + "\n")
    log_fp.close()