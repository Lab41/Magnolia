from flask import render_template, request, flash, send_file, redirect
from app import app
import numpy as np
from python_speech_features import sigproc
from keras.models import load_model
from python_speech_features.sigproc import deframesig
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import logging
import matplotlib.pylab as plt
import pylab
import collections
import os
from .tflow_functions import tflow_separate


project_root = app.root_path

model_path = project_root + '/static/overfitted_dnn_mask.h5'

ALLOWED_EXTENSIONS = set(['wav'])

app.config['SECRET_KEY'] = 'development key'

nfilt=64
numcep=64
nfft=512
winlen=0.01
winstep=0.005
ceplifter=0
fs = 16000

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


state = {'wav_list':[],'spec_file':None, 'input_signal_url':'/static/mixed_signal.wav'}

@app.route('/',methods = ['GET','POST'])
@app.route('/index.html',methods = ['GET','POST'])
def index():
    
    recon_signal = 1
    input_signal_filename = os.path.splitext(os.path.basename(state['input_signal_url']))[0]
    
    if request.method == 'POST' and request.form['btn'] == 'Visualize':
        fs,noisy_signal = wav.read(project_root + state['input_signal_url'])
        mfcc_feat = mfcc_feature_extractor(noisy_signal)
        plot_spectogram(mfcc_feat,project_root + '/resources/spec_'+ input_signal_filename + '.png')
        state['spec_file'] = 'spec_' + input_signal_filename + '.png'


    elif request.method == 'POST' and request.form['btn'] == 'Separate':
        fs,noisy_signal = wav.read(project_root + state['input_signal_url'])
        mfcc_feat = mfcc_feature_extractor(noisy_signal)
        mag,phase = feature_extractor(noisy_signal)
        mask = mask_prediction(model_path,mfcc_feat)
        recon_signal = (signal_reconstruction(mask,mag,phase)).astype(np.int16)

        wav.write(project_root + '/resources/' + input_signal_filename + 'split1.wav',fs,recon_signal)
        wav.write(project_root + '/resources/'+ input_signal_filename + 'split2.wav',fs,recon_signal)
      
        state['spec_file'] = 'spec_' + input_signal_filename + '.png'
        state['wav_list'] = [input_signal_filename + 'split1.wav',input_signal_filename + 'split2.wav']

    elif request.method == 'POST' and request.form['btn'] == 'Tflow_Separate':  
        
        #Separate speakers 
        signals = tflow_separate(project_root + state['input_signal_url'])
        state['wav_list'][:] = [] 
        for index,speaker in enumerate(signals):
            wav.write(project_root + '/resources/' + input_signal_filename + 'tflowsplit'+str(index)+'.wav',10000,speaker) 
            state['wav_list'].append(input_signal_filename + 'tflowsplit'+str(index)+'.wav')

        state['spec_file'] = 'spec_' + input_signal_filename + '.png'    

    return render_template('index.html',
                           title='Home',
                           wav_files=state['wav_list'],
                           input_sound=state['input_signal_url'],
                           input_spec_url=state['spec_file'],
                           )

@app.route('/upload',methods = ['POST'])
def upload():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect('index.html')
    upload_file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if upload_file.filename == '':
        flash('No selected file')
        return redirect('index.html')
    if upload_file and allowed_file(upload_file.filename): 
        app.logger.info('In the upload')   

        upload_file.save(project_root + '/resources/' + upload_file.filename)
        state['input_signal_url'] = '/resources/' + upload_file.filename

    return render_template('index.html',
                           title='Home',
                           wav_files=state['wav_list'],
                           input_sound=state['input_signal_url'],
                           input_spec_url=state['spec_file'],
                           )


@app.route('/resources/<string:file_name>',methods = ['GET','POST'])
def resources(file_name):
    app.logger.info('In the resources')
    return send_file(project_root + '/resources/'+ file_name)

def specdecomp(signal,samplerate=16000,winlen=0.025,winstep=0.01,
              nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
              winfunc=lambda x:np.ones((x,)),decomp='complex'):

    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1). 
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """    
    
    highfreq= highfreq or samplerate/2
    signal = sigproc.preemphasis(signal,preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    if decomp=='time' or decomp=='frames':
        return frames
    
    complex_spec = np.fft.rfft(frames,nfft)    
    if decomp=='magnitude' or decomp=='mag' or decomp=='abs':
        return np.abs(complex_spec)
    elif decomp=='phase' or decomp=='angle':
        return np.angle(complex_spec)
    elif decomp=='power' or decomp=='powspec':
        return sigproc.powspec(frames,nfft)
    else:
        return complex_spec        
    return spect

def plot_spectogram(features,file_path):
    plt.clf()
    plt.imshow(np.sqrt(features.T) , origin='lower' ,cmap='bone_r')
    plt.savefig(file_path)


def mfcc_feature_extractor(noisy_signal):
   
    mfcc_feat = mfcc(noisy_signal,fs,nfilt=nfilt,numcep=numcep,nfft=nfft,
                 winlen=winlen,winstep=winstep,ceplifter=ceplifter,
                 appendEnergy=False)
    return mfcc_feat


def feature_extractor(noisy_signal):
    
    mfcc_magni = specdecomp(noisy_signal,samplerate=fs,nfft=nfft,
                        winlen=winlen,winstep=winstep,decomp='abs')
    mfcc_phase = specdecomp(noisy_signal,samplerate=fs,nfft=nfft,
                        winlen=winlen,winstep=winstep,decomp='phase')

    return mfcc_magni,mfcc_phase

def mask_prediction(model_path, signal_features):

    model = load_model(model_path)
    mask = model.predict(signal_features)

    return mask

def signal_reconstruction(mask,mag,phase):

    recon_signal = (mask * mag) * np.exp( 1j *  phase)
    recon_signal = np.fft.irfft(recon_signal)
    recon_signal = recon_signal[:,:(int(fs*winlen))]
    recon_signal = deframesig(recon_signal, 0, int(fs*winlen), int(fs*winstep))

    return recon_signal


