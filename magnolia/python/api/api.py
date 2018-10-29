import json
import os
import tempfile
import logging.config
from flask import Flask, request, jsonify, url_for, send_from_directory
from flask_cors import CORS, cross_origin
import librosa
import numpy as np

from magnolia.preprocessing.preprocessing import normalize_waveform
from magnolia.preprocessing.preprocessing import preprocess_waveform
from magnolia.preprocessing.preprocessing import undo_preprocessing
from magnolia.utils.postprocessing import convert_preprocessing_parameters
from magnolia.utils.clustering_utils import chimera_clustering_separate, chimera_mask
from magnolia.models import make_model

UPLOAD_DIR = './uploads'
CONVERTED_DIR = './converted'
MODEL_DIR = './model'
MODEL_DEVICE = '/cpu:0'
ALLOWED_EXTENSIONS = set(['wav'])
SAMPLING_RATE = 16000

settings={
    "compression": "gzip",
    "compression_opts": 0,
    "processing_parameters": {
        "target_sample_rate": 10000,
        "preemphasis_coeff": 0.0,
        "stft_args": {
            "n_fft": 512,
            'hop_length': 256
        }
    }
}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_DIR'] = UPLOAD_DIR
app.config['CONVERTED_DIR'] = CONVERTED_DIR
app.config['MODEL_DIR'] = MODEL_DIR

def valid_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_temp_file(request):
    if 'file' not in request.files:
        flash('No file part')
        return ''
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return ''
    if file and valid_file(file.filename):
        tf = tempfile.NamedTemporaryFile()
        filename = os.path.basename(tf.name)
        temp_path = os.path.join(app.config['UPLOAD_DIR'], filename)
        file.save(temp_path)
        return temp_path


def convert_file_to_waveform(file_path):
    y,sr = librosa.load(file_path)
    print("sampling rate: {0}".format(sr))
    duration = librosa.get_duration(y=y, sr=sr)
    print("durations: {0}".format(duration))
    print("len in bits: {0}".format(int(duration*sr)))
    return y, sr, duration

def prepare_waveform(wf_batch):
    # Compute STFT spectrogram
    signal = wf_batch #normalize_waveform(np.squeeze(wf_batch))
    D,yy = preprocess_waveform(wf_batch, SAMPLING_RATE, **settings['processing_parameters'])
    app.logger.debug(D.shape)
    app.logger.debug(yy.shape)
    return D

def process_waveform(D):
    frequency_dim = D.shape[0]
    print("D shape 1: {0}".format(D.shape[1]))
    model_save_base = MODEL_DIR
    model_location = MODEL_DEVICE

    model_params = {
            'layer_size': 500,
            'embedding_size': 10,
            'alpha': 0.1,
            'nonlinearity': 'tf.tanh',
        }
    model_params['F'] = frequency_dim
    config = {'model_params': model_params,
              'device': model_location}
    model = make_model('Chimera', config)
    model.load(model_save_base)

    source_specs = chimera_mask(np.expand_dims(D,axis=0), model)[0]
    print(source_specs[0].shape)
    return source_specs

def postprocess_waveform(source_specs, sample_length_in_bits):
    istft_args={'hop_length': 256, 'win_length': 512}
    yy_out = undo_preprocessing(source_specs[:,:,0], sample_length_in_bits,
                                   preemphasis_coeff=0,
                                   istft_args=istft_args)

    return yy_out

def convert_back_to_wav(yy, save_sr):
    tf = tempfile.NamedTemporaryFile()
    filename = os.path.basename(tf.name)
    temp_path = os.path.join(app.config['CONVERTED_DIR'], filename)
    librosa.output.write_wav(temp_path, yy, save_sr)
    return url_for('uploaded_file', filename=filename)

@app.route('/api/v1/converted/<filename>', methods=['GET'])
@cross_origin(origin='*')
def uploaded_file(filename):
    return send_from_directory(app.config['CONVERTED_DIR'],
                               filename, attachment_filename=filename + '.wav', mimetype='audio/wav')

@app.route('/api/v1/convert', methods=['POST'])
@cross_origin(origin='*')
def convert_file():
    temp_path = save_temp_file(request)
    wf_batch, load_sr, len_in_secs = convert_file_to_waveform(temp_path)
    SAMPLING_RATE = load_sr
    len_in_bits = int(16000*len_in_secs)
    D = prepare_waveform(wf_batch)
    source_specs = process_waveform(D)
    yy = postprocess_waveform(source_specs, len_in_bits)
    converted_url = convert_back_to_wav(yy, 13000)
    return converted_url

if __name__ == '__main__':
    app.logger = logging.getLogger('api')
    app.logger.debug("logging started")
    app.run(port='5001', debug=True)