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
import soundfile as sf
from .keras_functions import keras_separate,keras_spec

project_root = app.root_path

ALLOWED_EXTENSIONS = set(['wav'])

app.config['SECRET_KEY'] = 'development key'


state = {'wav_list':[],'spec_file':None, 'input_signal_url':None}

@app.route('/',methods = ['GET','POST'])
@app.route('/index.html',methods = ['GET','POST'])
def index():
    
    recon_signal = 1
    if(state['input_signal_url'] != None):
        input_signal_filename = os.path.splitext(os.path.basename(state['input_signal_url']))[0]
    
    if request.method == 'POST' and request.form['btn'] == 'Separate' and state['input_signal_url'] != None :
        app.logger.info('In the separate')
        #Separate speakers 
        signals = keras_separate(project_root + state['input_signal_url'],project_root+'/static/overfitted_dnn_mask.h5')
        state['wav_list'][:] = [] 
        for index,speaker in enumerate(signals): 
            sf.write(project_root + '/resources/' + input_signal_filename + 'kerassplit'+str(index)+'.wav',speaker,16000) 
            state['wav_list'].append(input_signal_filename + 'kerassplit'+str(index)+'.wav')
  

    elif request.method == 'POST' and request.form['btn'] == 'Tflow_Separate' and state['input_signal_url'] != None:  
        
        #Separate speakers 
        signals = tflow_separate(project_root + state['input_signal_url'])
        state['wav_list'][:] = [] 
        for index,speaker in enumerate(signals): 
            sf.write(project_root + '/resources/' + input_signal_filename + 'tflowsplit'+str(index)+'.wav',speaker,10000) 
            state['wav_list'].append(input_signal_filename + 'tflowsplit'+str(index)+'.wav')
  

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

    def allowed_file(filename):
        return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    if upload_file and allowed_file(upload_file.filename): 
        app.logger.info('In the upload')   

        upload_file.save(project_root + '/resources/' + upload_file.filename)
        state['input_signal_url'] = '/resources/' + upload_file.filename

        #Plot spectogram with uploaded input file
        input_signal_filename = os.path.splitext(os.path.basename(state['input_signal_url']))[0] 

        #features = keras_spec(project_root + state['input_signal_url'])
        f,t,Sxx = keras_spec(project_root + state['input_signal_url'])

        #plot_spectogram(features,project_root + '/resources/spec_'+ input_signal_filename + '.png')
        plot_spectogram(f,t,Sxx,project_root + '/resources/spec_'+ input_signal_filename + '.png')

        state['spec_file'] = 'spec_' + input_signal_filename + '.png'


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


def plot_spectogram(f,t,Sxx,file_path):
    #plt.clf()
    #plt.imshow(np.sqrt(features.T) , origin='lower' ,cmap='bone_r')
    plt.pcolormesh(t, f, Sxx, cmap='bone_r')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig(file_path)


