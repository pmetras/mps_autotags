
# mps_autotags
# ============
# Generate automatic keywords from features recognized in pictures by a neural network and save them into
# MyPhotoShare 'album.ini' metadata files.
#
# Author: Pierre Métras
# Date: 20200329

import sys
import os
import argparse
import json
import csv
import zipfile
import time
import string
import pickle
import random
import ast
import configparser
import math

import psutil

import numpy as np

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize

# Disable Tensorflow warnings
#TODO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# https://github.com/tensorflow/tensorflow/issues/33022
import tensorflow as tf

import tensorflow.keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.xception import Xception, preprocess_input as xception_preprocess_input


# Global constants
DATADIR = os.path.normpath(os.path.join(os.path.realpath(__file__), "../../data/data"))
VGENOMEDIR = os.path.normpath(os.path.join(os.path.realpath(__file__), "../../data/vgenome"))
MSCOCODIR = os.path.normpath(os.path.join(os.path.realpath(__file__), "../../data/mscoco"))
NNDIR = os.path.normpath(os.path.join(os.path.realpath(__file__), "../../data/nn"))

VGENOME_SYNSETS = "synsets.json"
VGENOME_IMAGES = "image_data.json"
VGENOME_OBJECTS = "objects.json"
VGENOME_ATTRIBUTES = "attributes.json"
VGENOME_RELATIONSHIPS = "relationships.json"

COCO_ANNOTATIONS_ZIP = "annotations_trainval2017.zip"
COCO_CAPTIONS_TRAIN = "annotations/captions_train2017.json"
COCO_CAPTIONS_VAL = "annotations/captions_val2017.json"


SYNSETS_CSV = "synsets.csv"
IMAGES_CSV = "images.csv"

CONFIG_INI = "config.ini"

# Punctuation translation table
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
# English stopwords
STOPWORDS = set(stopwords.words('english'))
# WordNet lemmatizer
LEMMATIZER = WordNetLemmatizer()

# Image features dimensions
TARGET_SIZE = (224, 224)
# Model input shape
INPUT_SHAPE = (224, 224, 3)


# Global variables
data_dir = DATADIR
vgenome_dir = VGENOMEDIR
mscoco_dir = MSCOCODIR
nn_dir = NNDIR

# Dictionary of synsets. `reversed_synsets` is only initialized
# when the initial list is built.
synsets = {}
reversed_synsets = {}

# Preprocessing for the current model
preprocess_input = None


def main():
    """
    The main dispatcher.
    mps_autotags is run in 3 phases:
        1. Prepare the datasets, pictures and vocabulary (synsets).
        2. Train a neural network to recognize features in pictures.
        3. Let the neural network recognize features in a new set of pictures and generate the keywords.
    """
    parser = argparse.ArgumentParser(description="Automatic keywords indexer for MyPhotoShare",
            usage="""mps_autotags command [--options]
Automatically create keywords from pictures album content, for use
in MyPhotoShare gallery.

Commands can be:
    init        Create environment from Visual Genome and MS COCO files
    learn       Train neural network
    generate    Generate keywords for pictures in an album directory

Type `mps_autotags command --help` for command options.

After the optional initial initialization and learning phase, one will run from
time to time `mps_autotags generate --album-dir=PATH --language=fra` to
generate new keywords for new photo album.
""")
    parser.add_argument("command", help="The action to run")

    args = parser.parse_args(sys.argv[1:2])
    # Call the command
    globals()[args.command]()


def log(msg):
    """
    Print a message.
    """
    print("{} - {}".format(int(time.process_time()), msg))


def logerr(msg):
    """
    Print an error message.
    """
    print("{} - ERROR: {}".format(int(time.process_time()), msg), file=sys.stderr)


def load_config():
    """
    Load the values in the 'config.ini' file before starting a new phase.
    """
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(os.path.join(DATADIR, CONFIG_INI))

    global data_dir, vgenome_dir, mscoco_dir, nn_dir
    data_dir = config['DEFAULT']["data_dir"]
    vgenome_dir = config['DEFAULT']["vgenome_dir"]
    mscoco_dir = config['DEFAULT']["mscoco_dir"]
    nn_dir = config['DEFAULT']["nn_dir"]

    return config


def set_if_value(config, section, option, value=None):
    """
    Save a value in the 'config.ini' file depending on the existing values
    in that file.

    Initial     Param       Result
                V           V
    V           None
                None
    V1          V2          V2
    """
    if value == None:
        if config.has_option(section, option):
            config.remove_option(section, option)
    else:
        config.set(section, option, str(value))


def save_config(config, model_name, nb_synsets=None, augment=None, limit_pictures=None, max_epochs=None, val_percent=None, model_filename=None):
    """
    Save the state of a mps_autotags running phase into a 'config.ini' file to keep the results
    fro the next phase.
    """
    if config is None:
        config = configparser.ConfigParser()
        config.optionxform = str

    config['DEFAULT']["data_dir"] = data_dir
    config['DEFAULT']["vgenome_dir"] = vgenome_dir
    config['DEFAULT']["mscoco_dir"] = mscoco_dir
    config['DEFAULT']["nn_dir"] = nn_dir
    config['DEFAULT']["current_model"] = model_name
    config['DEFAULT']["nb_synsets"] = str(nb_synsets)

    if model_name not in config.sections():
        config.add_section(model_name)

    set_if_value(config, model_name, "model_filename", model_filename)
    set_if_value(config, model_name, "augment", augment)
    set_if_value(config, model_name, "limit_pictures", limit_pictures)
    set_if_value(config, model_name, "max_epochs", max_epochs)
    set_if_value(config, model_name, "val_percent", val_percent)

    with open(os.path.join(DATADIR, CONFIG_INI), 'w') as configfile:
        config.write(configfile)

    return config


def init():
    """
    This is the first phase of the mps_autotags process: we prepare sets of pictures from Visual Genome and
    MS COCO to extract main features. At the same time, we generate a vocabulary of terms that will be used
    for the training phase.
    """
    log("Initializing environment from Visual Genome and MS COCO files")
    global data_dir, vgenome_dir, mscoco_dir
    parser = argparse.ArgumentParser(description="Create environment from Visual Genome and MS COCO files",
        usage="""mps_autotags init [--options]
Prepare mps_autotags environment to train a neural network to recognize features from pictures.
Pictures dataset are taken from Visual Genome and MS COCO datasets.
Keywords are learnt from Visual Genome Wordnet picture descriptions and MS COCO captions.

One can generate artificial pictures from the original datasets with the '--augment=N' option, giving more
training data to the neural network for better results but also slowing the training process by a factor of N.
""")
    parser.add_argument("--data-dir", default=data_dir, help="Directory where mps_autotags saves its data files (default=data/data)")
    parser.add_argument("--vgenome-dir", default=vgenome_dir, help="Directory with the training image data from Visual Genome (default=data/vgenome)")
    parser.add_argument("--mscoco-dir", default=mscoco_dir, help="Directory with the training image data from MS COCO (default=data/mscoco)")
    parser.add_argument("--model", default="xception", help="Pretrained neural network model architecture (default=xception)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--augment', type=int, default=1, help="Data augmentation: number of augmented image variants created (default=1)")
    group.add_argument("--no-features", action='store_true', help="Don't extract features from images")

    parser.add_argument("--add-attributes", action='store_true', help="Add keywords from Visual Genome objects attributes")
    parser.add_argument("--add-relations", action='store_true', help="Add keywords from Visual Genome objects relationships")
    parser.add_argument("--add-synsets", action='store_true', help="Add synsets found in MS COCO captions even if they don't exist in Visual Genome")
    parser.add_argument("--limit-pictures", type=int, default=-1, help="Limit the number of pictures to extract features from, useful for debugging")

    args = parser.parse_args(sys.argv[2:])

    # User defined data directories
    data_dir = check_dir(args.data_dir)
    vgenome_dir = check_dir(args.vgenome_dir)
    mscoco_dir = check_dir(args.mscoco_dir)

    if args.model not in ['xception', 'vgg16']:
        logerr("--model: Pretrained neural network value must be chosen in ['xception', 'vgg16']")
        quit(1)

    if args.limit_pictures > 0 and args.limit_pictures < 100:
        logerr("--limit-pictures: Value must be >= 100")
        quit(1)

    mem = psutil.virtual_memory()
    if mem.total < 16 * 1024 * 1024 * 1024:
        log("SOME MODELS NEED MORE THAN 16 GB OF MEMORY! BE PREPARED TO SUFFER...")

    # Build synsets and images
    build_synsets_list()
    build_images_list(add_attributes=args.add_attributes, add_relations=args.add_relations, add_synsets=args.add_synsets)
    nb_synsets = save_synsets_list()

    # Extract features from images
    if not args.no_features:
        prepare_images(args.model, args.augment, args.limit_pictures)

    # Save config
    config = load_config()
    save_config(config, args.model, nb_synsets=nb_synsets, augment=args.augment, limit_pictures=args.limit_pictures)


def learn():
    """
    This is the second phase of the mps_autotags process: we train a neural network to recognize items in pictures.

    The training pictures are taken from Visual Genome and MX COCO datasets and are associated with captions or
    descriptions.
    """
    log("Learning keywords from images set")
    global data_dir, nn_dir
    parser = argparse.ArgumentParser(description="Learn from image data",
        usage="""mps_autotags learn [--options]
Train a neural network to associate caption keywords with image features that were prepared during the 'init' phase.
The neural network will be trained for a maximum of '--max-epochs' epochs with gradient descent.
'--val-percent' percents of the pictures will be used to validate and guide the training process.
""")
    parser.add_argument("--max-epochs", type=int, default=30, help="Maximum number of epochs (default=30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Size of training batches (default=32)")
    parser.add_argument("--val-percent", type=int, default=10, help="Percentage of training images used for validation (default=10%%)")

    args = parser.parse_args(sys.argv[2:])

    config = load_config()
    model_name = config['DEFAULT']["current_model"]
    # We need to know the number of classes
    nb_synsets = int(config['DEFAULT']["nb_synsets"])

    augment = config.getint(model_name, "augment", fallback=1)
    limit_pictures = config.getint(model_name, "limit_pictures", fallback=-1)

    # User defined directories
    data_dir = check_dir(data_dir)
    nn_dir = check_dir(nn_dir)

    if args.val_percent < 1 or args.val_percent > 50:
        logerr("--val-percent: Validation images percentage must be between 1% and 50% of training set")
        quit(1)

    if args.batch_size < 1 or args.batch_size > 10000:
        logerr("--batch-size: The training batch value must be in the range 1 to 10000")
        quit(1)

    if model_name not in ['xception', 'vgg16']:
        logerr("--model: Pretrained neural network value must be chosen in ['xception', 'vgg16'] in 'init' command")
        quit(1)

    features_pkl = os.path.join(data_dir, "features-{}{}-{}.pkl".format(model_name, limit_pictures, augment))
    if not os.path.isfile(features_pkl):
        logerr("Prepared pictures feature file {} does not exist. You must prepare it with the 'init' command before using 'learn'.".format(features_pkl))
        quit(1)

    if limit_pictures == -1:
        max_features = -1
    else:
        max_features = limit_pictures * augment

    model = build_model(model_name, nb_synsets)
    model_filename = model_name + ".h5"
    train_model(model_filename, model, features_pkl, nb_synsets, batch_size=args.batch_size, max_epochs=args.max_epochs,
        val_percent=args.val_percent, max_features=max_features)

    save_config(config, model_name, nb_synsets=nb_synsets, augment=augment, limit_pictures=limit_pictures,
        max_epochs=args.max_epochs, val_percent=args.val_percent, model_filename=model_filename)


def generate():
    """
    This is the third phase of the mps_autotags process: keywords are generated by the trained neural network
    for a set of new pictures.

    Generate "auto_tags" values from a set of pictures, either from the content of a directory or a
    MyPhotoShare 'album.ini' file. If an 'album.ini' file is present in the directory, it will be used
    to select the pictures, else all pictures will be used.
    
    The "auto_tags" can be saved into the 'album.ini' file, under each picture metadata, or displayed on screen.
    CAUTION: When the 'album.ini' file is updated, comments are lots and layout of the file can be changed.
    That's a limitation of Python configparser file management.
    """
    log("Generate")
    global data_dir, nn_dir
    parser = argparse.ArgumentParser(description="Generate keywords",
        usage="""mps_autotags generate --lang=<ISO-639-3> [--options]
Create or update exiting `album.ini` file in the specified location with automatic keywords
for each picture in the `album.ini` file or in the directory.

Either '--album-dir' or '--album-ini' must be specified. If `album-dir` is given and there
exists an `album.ini` file, the content of that file will be used to find the pictures instead
of the content of the album directory.

To prevent overwriting `album.ini` file, use '--no-update'.

By default, 5 keywords are generated for each picture. If you want to keep only pertinent
keywords, you can use '--min-proba' where the software will only keep keywords with a confidence
probability over the value given. It can happen that no keywords are generated.

Note that `--lang` code is a ISO-639 3 characters language code like:
    ['als', 'arb', 'bul', 'cat', 'cmn', 'dan', 'ell', 'eng', 'eus', 'fas', 'fin',
     'fra', 'glg', 'heb', 'hrv', 'ind', 'ita', 'jpn', 'nld', 'nno', 'nob', 'pol',
     'por', 'qcn', 'slv', 'spa', 'swe', 'tha', 'zsm']

""")
    parser.add_argument("--lang", default='eng', help="Language used to generate keywords (default=eng)")

    parser.add_argument("--nb-keywords", type=int, default=5, help="Number of keywords to generate (default=5)")
    parser.add_argument("--min-proba", type=float, default=0.0, help="Minimal probability to keep a keyword")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--album-dir", help="Directory containing the pictures that will be indexed")
    group.add_argument("--album-ini", help="Path to the 'album.ini' file containing the pictures that will be indexed")

    parser.add_argument("--no-update", action='store_true', help="Don't write generated keywords into `album.ini` file")

    args = parser.parse_args(sys.argv[2:])

    config = load_config()
    model_name = config['DEFAULT']["current_model"]
    model_filename = config[model_name]["model_filename"]

    # User defined directories
    data_dir = check_dir(data_dir)
    nn_dir = check_dir(nn_dir)

    if args.album_dir:
        pictures_dir = check_dir(args.album_dir)
        album_ini = 'album.ini'
    elif args.album_ini:
        if not os.path.isfile(args.album_ini):
            logerr("{} does not exist or is not a file".format(args.album_ini))
            exit(1)

        pictures_dir = os.path.dirname(args.album_ini)
        album_ini = os.path.basename(args.album_ini)
    else:
        logerr("Either '--album-dir' or '--album-ini' options must be specified.")
        exit(1)

    if args.min_proba < 0.0 or args.min_proba > 1.0:
        logerr("--min-proba: Minimal probability of selecting keywords must be in interval [0.0; 1.0]")
        quit(1)

    global synsets, reversed_synsets
    synsets, reversed_synsets = load_synsets_list()
    inifile = generate_keywords(album_ini, pictures_dir, model_name, model_filename, args.lang, min_proba=args.min_proba, nb_keywords=args.nb_keywords)

    # If the user wants to keep the updated `album.ini` file, write it to disk.
    if not args.no_update:
        album_path = os.path.join(pictures_dir, album_ini)
        log("Saving keywords into {}".format(album_path))
        with open(album_path, 'w') as album_file:
            inifile.write(album_file)
    else:
        print("== Generated Keywords ===")
        print("Picture filename --> Tags")
        print("-------------------------")
        for picture in inifile.sections():
            if 'auto_tags' in inifile[picture]:
                keywords = inifile[picture]['auto_tags']
                print(picture, "-->", keywords)
        print("-------------------------")


def add_synset(synset_name):
    """
    Add a synset to the dictionary of synsets and gives it a numeric identifier.
    """
    global synsets, reversed_synsets
    synset_id = len(synsets) + 1
    synsets[synset_id] = {'synset_id': synset_id, 'synset_name': synset_name}
    reversed_synsets[synset_name] = synset_id
    return synset_id


def build_synsets_list():
    """
    Convert the set of Wordnet synsets used in the Visual Genome project
    into a CVS file, giving a numeric identifier to each synset.
    Initialize the `reversed_synsets` dictionary that will be used to
    create the list of images.
    Return a tuple of dictionary of synsets.
    """
    log("Prepare synsets list")
    synsets_json = None
    with zipfile.ZipFile(os.path.join(vgenome_dir, VGENOME_SYNSETS + ".zip")) as synsets_zip:
        with synsets_zip.open(VGENOME_SYNSETS) as synsets_file:
            synsets_json = json.load(synsets_file)

    global synsets, reversed_synsets
    for synset in synsets_json:
        add_synset(synset['synset_name'])
    log("Loaded {} synsets".format(len(synsets)))
    del synsets_json

    return (synsets, reversed_synsets)


def save_synsets_list():
    """
    Save all the synsets in the vocabulary into a CSV file.
    """
    with open(os.path.join(data_dir, SYNSETS_CSV), 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['synset_id', 'synset_name'], dialect='unix')
        writer.writeheader()
        for synset_id in synsets.keys():
            writer.writerow(synsets[synset_id])

    nb_synsets = len(synsets)
    log("Saved {} synsets".format(nb_synsets))
    return nb_synsets


def load_synsets_list():
    """
    Load the CSV file of synsets into a dictionary.
    """
    log("Load synsets list")
    synsets = {}
    reversed_synsets = {}
    with open(os.path.join(data_dir, SYNSETS_CSV), newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            synsets[int(row['synset_id'])] = row
            reversed_synsets[row['synset_name']] = int(row['synset_id'])
    log("Loaded {} synsets".format(len(synsets)))

    return (synsets, reversed_synsets)


def keywords_from_synset_id(synset_id, lang):
    """
    Get the list of keywords associated to the synset identifier in the given
    language. The `lang` code uses ISO-639 language codes.
    Common values are:
    ['als', 'arb', 'bul', 'cat', 'cmn', 'dan', 'ell', 'eng', 'eus', 'fas', 'fin',
     'fra', 'glg', 'heb', 'hrv', 'ind', 'ita', 'jpn', 'nld', 'nno', 'nob', 'pol',
     'por', 'qcn', 'slv', 'spa', 'swe', 'tha', 'zsm']
    Returns a list of keywords. The list can be null or contain multiple keywords.
    """
    global synsets
    synset_name = synsets[synset_id]['synset_name']
    return [lemma.name() for lemma in wn.synset(synset_name).lemmas(lang=lang)]


def keyword_from_synset_id(synset_id, lang):
    """
    Get the first keyword associated to a synset identifier in the given language.
    Lemmas in synsets are sorted by how often they appear (in the corpus used to create Wordnet).
    Well, sort of... From the WordNet FAQ:
    "WordNet senses are ordered using sparse data from semantically tagged text. The order of the 
    senses is given simply so that some of the most common uses are listed above others (and those 
    for which there is no data are randomly ordered). The sense numbers and ordering of senses in 
    WordNet should be considered random for research purposes."
    """
    global synsets
    synset_name = synsets[synset_id]['synset_name']
    lemmas = wn.synset(synset_name).lemmas(lang=lang)

    if len(lemmas) > 0:
        return lemmas[0].name()
    else:
        return None


def synset_id_from_word(word, add_synsets=False):
    """
    Get the synset id from any word (or string).
    If a synset can't be found, return 0.
    """
    synset_id = 0
    synsets = wn.synsets(word)
    if len(synsets) > 0:
        synset = synsets[0]
        try:
            # If existing synset from Visual Genome 18000 synsets.
            synset_id = id_from_synset(synset.name())
        except:
            # Add the new synsets if the user wants it
            if add_synsets:
                synset_id = add_synset(synset.name())

    return synset_id


def id_from_synset(synset):
    """
    From a synset name (ie. "dog.n.01"), returns its identifier.
    Raise a KeyValue exception if the synset does not exist in the
    data set.
    """
    return reversed_synsets[synset]


def build_images_list(add_attributes=False, add_relations=False, add_synsets=False):
    """
    Build the CSV file containing all the image identifiers associated with
    their size and synsets identifiers contained in the picture.
    To do that, we look at the Visual Genome objects, attributes and relationships files
    and extract the synsets. We consider also the MS COCO captions when
    the `coco_id` is specified. That way, we provide 2 sets of synsets:
    general ones from captions, and detailed ones from picture objects.
    Note that not all pictures have a `coco_id`.
    By default, objects attributes and relationships are not added to the
    list of synsets of images. You need to set them explicitely with
    `add_attributes` abd `add_relations` parameters. As the Visual Genome
    images are very rich, the number of synsets in scenes can dilute the
    learning process.
    """
    log("Build images list")
    images = {}

    # Look at objects first to collect symsets ids.
    log("Extract objects")
    objects_json = None
    with zipfile.ZipFile(os.path.join(vgenome_dir, VGENOME_OBJECTS + ".zip")) as objects_zip:
        with objects_zip.open(VGENOME_OBJECTS) as objects_file:
            objects_json = json.load(objects_file)

    for obj in objects_json:
        image_id = obj['image_id']
        image = images.get(image_id, {'image_id': image_id, 'synset_ids': [], 'names': [],
                                      'coco_id': None, 'captions': [], 'captions_synset_ids': []})
        synset_ids = image['synset_ids']
        names = image['names']
        for o in obj['objects']:
            for s in o['synsets']:
                synset_ids.append(id_from_synset(s))
            for n in o['names']:
                names.append(n)
                si = synset_id_from_word(n, add_synsets)
                if si > 0:
                    synset_ids.append(si)
        image['synset_ids'] = synset_ids
        image['names'] = names
        images[image_id] = image
    log("Loaded {} objects".format(len(objects_json)))
    del objects_json

    # Enrich with attributes
    if add_attributes:
        log("Extract attributes")
        attributes_json = None
        with zipfile.ZipFile(os.path.join(vgenome_dir, VGENOME_ATTRIBUTES + ".zip")) as attributes_zip:
            with attributes_zip.open(VGENOME_ATTRIBUTES) as attributes_file:
                attributes_json = json.load(attributes_file)

        for attr in attributes_json:
            image_id = obj['image_id']
            image = images.get(image_id, {'image_id': image_id, 'synset_ids': [], 'names': [],
                                          'coco_id': None, 'captions': [], 'captions_synset_ids': []})
            synset_ids = image['synset_ids']
            names = image['names']
            for a in attr['attributes']:
                for s in a['synsets']:
                    synset_ids.append(id_from_synset(s))
                for n in a['names']:
                    names.append(n)
                    si = synset_id_from_word(n, add_synsets)
                    if si > 0:
                        synset_ids.append(si)
            image['synset_ids'] = synset_ids
            image['names'] = names
            images[image_id] = image
        log("Loaded {} attributes".format(len(attributes_json)))
        del attributes_json

    # Last add relationships
    # We don't look at the synsets of the subject and the object in the relation.
    if add_relations:
        log("Extract relationships")
        relations_json = None
        with zipfile.ZipFile(os.path.join(vgenome_dir, VGENOME_RELATIONSHIPS + ".zip")) as relations_zip:
            with relations_zip.open(VGENOME_RELATIONSHIPS) as relations_file:
                relations_json = json.load(relations_file)

        for rel in relations_json:
            image_id = obj['image_id']
            image = images.get(image_id, {'image_id': image_id, 'synset_ids': [], 'names': [],
                                          'coco_id': None, 'captions': [], 'captions_synset_ids': []})
            synset_ids = image['synset_ids']
            names = image['names']
            for r in rel['relationships']:
                for s in r['synsets']:
                    synset_ids.append(id_from_synset(s))
                for n in r['predicate']:
                    names.append(n)
                    si = synset_id_from_word(n, add_synsets)
                    if si > 0:
                        synset_ids.append(si)
            image['synset_ids'] = synset_ids
            image['names'] = names
            images[image_id] = image
        log("Loaded {} relations".format(len(relations_json)))
        del relations_json

    log("Extract COCO captions")
    captions_json = None
    captions = []
    with zipfile.ZipFile(os.path.join(mscoco_dir, COCO_ANNOTATIONS_ZIP)) as captions_zip:
        log("Read training captions")
        with captions_zip.open(COCO_CAPTIONS_TRAIN) as captions_file:
            captions_json = json.load(captions_file)
            captions.extend(captions_json['annotations'])
        log("Read validation captions")
        with captions_zip.open(COCO_CAPTIONS_VAL) as captions_file:
            captions_json = json.load(captions_file)
            captions.extend(captions_json['annotations'])
    log("Loaded {} captions".format(len(captions)))
    del captions_json

    # Then look at images to find the missing parameters.
    log("Extract images")
    images_json = None
    with zipfile.ZipFile(os.path.join(vgenome_dir, VGENOME_IMAGES + ".zip")) as images_zip:
        with images_zip.open(VGENOME_IMAGES) as images_file:
            images_json = json.load(images_file)

    coco_ids = {}
    for image in images_json:
        image_id = image['image_id']
        coco_id = image['coco_id']
        images[image_id]['width'] = image['width']
        images[image_id]['height'] = image['height']
        images[image_id]['coco_id'] = coco_id
        if coco_id is not None:
            coco_ids[coco_id] = image_id

    # Add COCO captions
    log("Addding COCO captions")
    for caption in captions:
        coco_id = caption['image_id']
        # Not all COCO images are in Visual Genome
        if coco_id in coco_ids:
            image_id = coco_ids[coco_id]
            cleaned_caption = clean_caption(caption['caption'])
            images[image_id]['captions'].append(cleaned_caption)

    # Convert COCO captions to synsets
    log("Converting captions to synsets")
    counter = 0
    for image_id, image in images.items():
        for caption in image['captions']:
            for word in caption.split():
                try:
                    # We look at the root name (drop plurials, inflectional ending rules, etc.
                    # Probably useless as captions have been NLTK cleaned...
                    # We consider only the first synset of that lemma to limit the number
                    # of synsets with similar meanings.
                    synset_id = synset_id_from_word(wn.morphy(word), add_synset)
                    if synset_id > 0:
                        counter += 1
                        if counter % 100000 == 0:
                            print(".", end='')
                            sys.stdout.flush()
                        images[image_id]['captions_synset_ids'].append(synset_id)
                except:
                    continue
    print("")
    print("Found {} synsets".format(counter))

    # Delete duplicates
    log("Delete duplicate synsets")
    for image_id, image in images.items():
        images[image_id]['synset_ids'] = list(set(image['synset_ids']))
        images[image_id]['names'] = list(set(image['names']))
        images[image_id]['captions_synset_ids'] = list(set(image['captions_synset_ids']))

    # Now we write the CSV, keeping the order it had in the images JSON
    # file to ease debugging.
    log("Saving image captions")
    with open(os.path.join(data_dir, IMAGES_CSV), 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['image_id', 'coco_id', 'width', 'height', 
                                                      'synset_ids', 'names', 'captions_synset_ids', 'captions'], dialect='unix')
        writer.writeheader()
        for image in images_json:
            writer.writerow(images[image['image_id']])
    del images_json

    return images


def check_dir(path):
    """
    Check that the path given is a valid directory.
    If not, print an error message and abort the program.
    Return the path.
    """
    if not path or not os.path.isdir(path):
        logerr("{} is not a valid directory".format(path))
        quit(1)
    return path


def clean_caption(caption):
    """
    Prepare caption string to extract synsets.
    * Remove punctuation
    * Remove extra white space
    * Convert to lower case
    * Remove stopwords
    * Remove words containing numbers
    """
    # Remove LF/CR
    s = caption.rstrip(string.whitespace)
    # Remove punctuation
    s = s.translate(PUNCT_TABLE)
    # Lower case
    s = s.lower()

    #s = word_tokenize(s)
    s = text_to_word_sequence(s)
    t = []
    for word in s:
        # Remove stop words
        if word in STOPWORDS:
            continue
        # Remove numbers and other characters
        if not word.isalpha():
            continue
        # Lemmatize
        w = LEMMATIZER.lemmatize(word, pos = "n")
        w = LEMMATIZER.lemmatize(w, pos = "v")
        w = LEMMATIZER.lemmatize(w, pos = ("a"))
        # We keep the word
        t.append(w)

    s = " ".join(t)
    
    return s


def load_images():
    """
    Load the CSV file of images into a dictionary.
    """
    log("Load images list")
    images = {}
    with open(os.path.join(data_dir, IMAGES_CSV), newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            images[int(row['image_id'])] = row
    log("Loaded {} images definitions".format(len(images)))

    return images


def select_model(model_name):
    """
    Prepare a model from a pretrained one. We can try as many different model
    types as we want.
    """
    log("Selecting model from parameters")
    global preprocess_input

    if model_name == 'xception':
        # Load Xception model
        model = Xception(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='avg')
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

        # For -1
        #output_shape = (2048, )
        # For -2
        #output_shape = (7, 7, 2048, )
        preprocess_input = xception_preprocess_input

    elif model_name == 'vgg16':

        # Load VGG16 model
        model = VGG16(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

        #output_shape = (4096, )
        preprocess_input = vgg16_preprocess_input
    else:
        logerr("Model name is not defined. Specify the model type you want to use.")
        quit(1)

    return model


def extract_images_features(model, img_dir1, img_dir2, augment=1, limit_pictures=-1):
    """
    Scan the directory given as parameter for image files and prepare
    them for the model, extracting their features. If `augment` is
    specified, does also random data augmentation:
      * Rotation up to 15 degrees
      * Heigh shift up to 10%
      * Width shift up to 10%
      * Horizontal flip
      * Zoom in the range [90%, 120%]
    Every 1000 images, dump the dictionary of features in the `features-partial.pkl`
    pickle file.
    When `limit_pictures` is specified, stops after this number of files (not pictures!)
    is processed.

    TODO: img_dir2 (=mscoco_dir / images) is not used presently as we train only on
    Visual Genome dataset.
    """
    log("Extract images features")

    # Random image generator
    datagen = None
    if augment > 1:
        datagen = ImageDataGenerator(
            rotation_range=15,              # Can rotate up to 15 degrees
            height_shift_range=0.1,         # Height shift of 10%
            width_shift_range=0.1,          # Width shift of 10%
            horizontal_flip=True,           # Horizontal flip is allowed
            brightness_range=[0.8, 1.2],    # Change brightness in range [80%, 120%]
            zoom_range=[0.9, 1.2])          # Zoom in range [90%, 120%]

        # Take into account augmentation into the number of pictures
        if limit_pictures != -1:
            limit_pictures = limit_pictures * augment

    # extract features from each photo
    features = dict()
    counter = 0
    for name in os.listdir(img_dir1):
        # Stop if the user does not want all pictures and we've reached the limit
        if limit_pictures != -1 and counter >= limit_pictures:
            break

        # load an image from file
        filename = os.path.join(img_dir1, name)
        image = None
        try:
            image = load_img(filename, target_size=TARGET_SIZE)
        except:
            logerr("File {} is not an image".format(name))
            continue

        # Do image augmentation
        for i in range(0, augment):
            img = image
            # Convert the image pixels to a numpy array
            img = img_to_array(img)

            # Augment data if requested
            if i != 0:
                img = datagen.random_transform(img)
            
            # Reshape data for the model
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

            # Prepare the image for the model: normalize data
            img = preprocess_input(img)

            # Predict features through image model
            img_feature = model.predict(img, verbose=0)

            # Store feature
            image_id = name.split('.')[0] + "-" + str(i)
            features[image_id] = img_feature
            log(">{} #{}".format(name, i))

            counter += 1
            if counter % 1000 == 0:
                log("Saving temporary image features: {} images processed".format(counter))
                with open(os.path.join(data_dir, "features-partial.pkl"), 'wb') as features_pkl:
                    pickle.dump(features, features_pkl)

    return features


def prepare_images(model_name, augment=1, limit_pictures=-1):
    """
    Extract the features of all training pictures and save them in the pickle file
    `features-{augment}.pkl`.
    """
    model = select_model(model_name)

    # Print model
    model.summary()

    # Extract features from all images in training directories
    features = extract_images_features(model, os.path.join(vgenome_dir, "images"), os.path.join(mscoco_dir, "images"), augment, limit_pictures)
    log("Extracted {} features".format(len(features)))

    # Final save to file
    with open(os.path.join(data_dir, "features-{}{}-{}.pkl".format(model_name, limit_pictures, augment)), 'wb') as features_pkl:
        log("Saving final image features")
        pickle.dump(features, features_pkl)

    if os.path.isfile(os.path.join(data_dir, "features-partial.pkl")):
        os.remove(os.path.join(data_dir, "features-partial.pkl"))


def load_prepared_images(features_pkl):
    """
    Load the features of pictures prepared during `init` phase.
    The `features_pkl` filename is the absolute path where pictures features
    have been preprocessed.
    """
    with open(features_pkl, 'rb') as features_file:
        features = pickle.load(features_file)
    
    return features


def build_model(model_name, nb_synsets):
    """
    Builds the multi-classification embeded model that will be trained.
    This model is a 2 dense layers after a 50% dropout layer.
    As we have > 20k synsets (targets) and we need to determine multiple ones in
    the pictures, we spread the output using binary cross-entropy with ADAM
    optimizer.
    """
    log("Building model")
    # As input, we'll use the preprocessed images
    model = select_model(model_name)
    input = Input(shape=model.output.shape.as_list()[1:])
    print("Output shape=", model.output.shape)

    flat1 = Flatten()(input)
    output1 = Dropout(0.5)(flat1)
    output2 = Dense(4096)(output1)
    output3 = Dense(8192)(output2)
    # We are using 'sigmoid' as we have multi-labels targets.
    output = Dense(nb_synsets, activation='sigmoid')(output3)

    model = Model(inputs=input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # Print model summary
    print(model.summary())

    return model


def use_generator(model_name, nb_synsets):
    """
    Do we need to use a data generator because there is not enough memory on the
    system? If less than 24 GB, we use generators instead of memory.
    """
    svmem = psutil.virtual_memory()
    if model_name == "vgg16":
        return svmem.total < 24_000_000_000
    elif model_name == "xception":
        return svmem.total < 16_000_000_000
    else:
        return svmem.total < 16_000_000_000


def train_model(model_name, model, features_pkl, nb_synsets, max_epochs=30, batch_size=32, val_percent=10, max_features=-1):
    """
    Train the model with gradient descent for `max_epochs` in batches of `batch_size`.
    Depending on the amount of memory available in the computer, the training can be done
    completely in memory or using batches of data created by generators.
    """
    log("Training model")

    # Generate checkpoint files so we can use the best model
    filepath = os.path.join(nn_dir, "model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Early stopping: Stop when validation loss is no more decreasing
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # TensorBoard to monitor learning
    tensorboard = TensorBoard(log_dir=nn_dir)

    if use_generator(model_name, nb_synsets):
        #TODO: This is the recommended way to use generators with Tensorflow 2.0.
        # Unfortunately, the API is too strict and does not support `dict` parameters.
        #DStrain, DSval = prepare_generator(features_pkl, nb_synsets, val_percent, max_features)
        #
        #log("Fit model with generator")
        #model.fit(x=DStrain, epochs=max_epochs, verbose=2, callbacks=[checkpoint, early_stopping], validation_data=DSval)

        # Doing it the old way...
        images = load_images()
        features = load_prepared_images(features_pkl)
        train_keys, val_keys = train_val_keys(features, val_percent, max_features)
        train_steps = math.ceil(len(train_keys) / batch_size)
        val_steps = math.ceil(len(val_keys) / batch_size)

        train_generator = data_generator(train_keys, features, images, nb_synsets, batch_size, type="t")
        val_generator = data_generator(val_keys, features, images, nb_synsets, batch_size, type="v")

        log("Fit model from generator")
        model.fit(x=train_generator, steps_per_epoch=train_steps, epochs=max_epochs, verbose=2,
                callbacks=[checkpoint, early_stopping, tensorboard], validation_data=val_generator, validation_steps=val_steps)
    else:
        Xtrain, ytrain, Xval, yval = prepare_data(features_pkl, nb_synsets, val_percent, max_features)

        # fit model
        log("Fit model")
        model.fit(x=Xtrain, y=ytrain, batch_size=batch_size, epochs=max_epochs, verbose=2,
                callbacks=[checkpoint, early_stopping, tensorboard], validation_data=(Xval, yval))

    log("Saving final model")
    model.save(os.path.join(nn_dir, model_name))


def train_val_keys(features, val_percent, max_features=-1):
    """
    Split the set of features into a traing and a validation set according
    to the percentage in `val_percent`.
    If `max_features` is specified, limit the number of features to that value.
    """
    if max_features == -1:
        nb_features = len(features)
    else:
        nb_features = max_features
    keys = random.sample(features.keys(), nb_features)
    
    nb_train = int(nb_features * (100 - val_percent) / 100)
    keys_train = random.sample(keys, nb_train)

    set_train = set(keys_train)
    keys_val = [item for item in keys if item not in set_train]

    return keys_train, keys_val


def data_generator(keys, features, images, nb_synsets, batch_size=32, type="."):
    """
    Create a generator preparing batch of data. This generator is used when all
    the dataset can't fit in memory. The data is taken from the set of `keys` and is
    returned in batches of `batch_size`. When all the keys have been selected, the
    generator starts again. Keys are randominzed first.
    The `type` value is printed on the console every time a batch is generated and
    can be used to distinguish training generator from the validation one.
    """
    log("Selecting training and validation dataset through generator")
    print(type.upper(), end='', flush=True)
    counter = 0
    X, y, i = [], [], 0
    while True:
        # Add a bit of randomness...
        random.shuffle(keys)

        for key in keys:
            counter += 1
            if counter % 10000 == 0:
                print()
                log("{} images processed in generator".format(counter))

            image = features[key]
            image_id = int(key.split('-')[0])
            try:
                synset_classes = best_synsets(images[image_id], nb_synsets)
                X.append(image[0])
                y.append(synset_classes)

                if i < batch_size:
                    i += 1
                else:
                    yield (np.asarray(X), np.asarray(y))
                    print(type, end='', flush=True)
                    X, y, i = [], [], 0
            except ValueError:
                #logerr("Not considering image[{}] without synsets".format(image_id))
                pass
            except KeyError:
                logerr("BUG! {} should be present in images".format(image_id))


def tf_data_generator(features, keys, images, nb_synsets):
    """
    A generator to create a dataset, to be used with `tf.data.Dataset.from_generator`.
    Unfortunately, this function can't understand the dictionary arguments of
    this generator!
    """
    for key in keys:
        image = features[key]
        image_id = int(key.split('-')[0])
        try:
            synset_classes = best_synsets(images[image_id], nb_synsets)
            #yield (np.asarray([image[0]]), np.asarray([synset_classes]))
            yield (image[0], synset_classes)
        except ValueError:
            logerr("Not considering image[{}] without synsets".format(image_id))
        except KeyError:
            logerr("BUG! {} should be present in images".format(image_id))


def prepare_generator(features_pkl, nb_synsets, val_percent=10, max_features=-1):
    """
    This function can't be used, though it is the recommended way to do it with
    Tensorflow 2. The reason is that the `args` argument of the `tf.data.Dataset.from_generator`
    call has to be converted to `tf.Tensor` objects, but this can't be done
    with `dict`.
    I keep the code in case it can be adapted in the future.
    """
    log("Selecting training and validation datasets through generator")
    images = load_images()
    features = load_prepared_images(features_pkl)
    keys_train, keys_val = train_val_keys(features, val_percent, max_features)

    DStrain = tf.data.Dataset.from_generator(tf_data_generator, args=[features, keys_train, images, nb_synsets],
        output_types=(tf.float32, tf.int32),
        output_shapes=((None, ), (nb_synsets, )))

    DSval = tf.data.Dataset.from_generator(tf_data_generator, args=[features, keys_val, images, nb_synsets],
        output_types=(tf.float32, tf.int32),
        output_shapes=((None, ), (nb_synsets, )))

    return DStrain, DSval
    

def prepare_data(features_pkl, nb_synsets, val_percent=10, max_features=-1):
    """
    Prepare the data from the training into a training and a validation datasets.
    A maximum of `max_features` are selected and then this number is split with
    `val_percent` put aside for the validation dataset.
    The `features_pkl` is the name of the pickled file containing the prepared
    images.
    For each image in the dataset, its set of synsets will be selected with the
    `best_synsets()` call, either from Visual Genome descriptions of MS COCO captions.
    Returns the tuple `Xtrain`, `ytrain`, `Xval`, `yval`.
    """
    log("Selecting training and validation dataset")
    images = load_images()
    features = load_prepared_images(features_pkl)
    keys_train, keys_val = train_val_keys(features, val_percent, max_features)

    # Prepare the training set
    Xtrain, ytrain = [], []
    for key in keys_train:
        image = features.pop(key)
        image_id = int(key.split('-')[0])
        try:
            # Don't use `images.pop(image_id)` because of augmentation, there can be multiple
            # images in `features` with the same `image_id`.
            synset_classes = best_synsets(images[image_id], nb_synsets)
            Xtrain.append(image[0])
            ytrain.append(synset_classes)
        except ValueError:
            logerr("Not considering image[{}] without synsets".format(image_id))
        except KeyError:
            logerr("BUG! {} should be present in images".format(image_id))

    Xtrain = np.asarray(Xtrain)
    ytrain = np.asarray(ytrain)

    print("Xtrain.shape=", Xtrain.shape)
    print("ytrain.shape=", ytrain.shape)

    # Prepare the validation dataset
    Xval, yval = [], []
    for key in keys_val:
        image = features.pop(key)
        image_id = int(key.split('-')[0])
        try:
            synset_classes = best_synsets(images[image_id], nb_synsets)
            Xval.append(image[0])
            yval.append(synset_classes)
        except ValueError:
            logerr("Not considering image[{}] without synsets".format(image_id))
        except KeyError:
            logerr("BUG! {} should be present in images".format(image_id))

    Xval = np.asarray(Xval)
    yval = np.asarray(yval)

    print("Xval.shape=", Xval.shape)
    print("yval.shape=", yval.shape)

    return Xtrain, ytrain, Xval, yval


def best_synsets(image, nb_synsets):
    """
    Select the best synsets representing a picture.
    * From MS COCO caption if available
    * From Visual Genome
    Visual Genome images can be too rich visually. Perhaps we want to capture
    only the most important objects from the picture. These objets are usually
    described in a caption.
    """
    synset_ids = None
    # 1. TODO: This piece of code favor using captions from COCO
    #if image['captions_synset_ids'] != "[]":
    #    synset_ids = ast.literal_eval(image['captions_synset_ids'])
    #else:
    #    synset_ids = ast.literal_eval(image['synset_ids'])

    # 2. TODO: This piece of code uses synsets from Visual Genome
    synset_ids = ast.literal_eval(image['synset_ids'])

    if synset_ids == []:
        raise ValueError(image['image_id'])

    # Make it multi-labels
    synset_ids = set(synset_ids)
    synset_classes = np.zeros((nb_synsets), dtype=int)
    for i in range(nb_synsets):
        if i in synset_ids:
            synset_classes[i] = 1

    return synset_classes


def find_pictures(album_ini, pictures_dir):
    """
    Prepare a configfile of all the picture names to index, from either
    the 'album.ini' file or the content of a directory.
    """
    # If the file already exist, use it
    album_file = os.path.join(pictures_dir, album_ini)
    if os.path.isfile(album_file):
        inifile = configparser.ConfigParser(allow_no_value=True)
        inifile.optionxform = str
        inifile.read(album_file)
    else:
        pictures = { os.path.join(pictures_dir, picture) : [] for picture in os.listdir(pictures_dir) }
        inifile = create_album_ini(pictures)

    return inifile


def generate_keywords(album_ini, pictures_dir, model_name, model_filename, lang, min_proba=0.0, nb_keywords=5):
    """
    Generate the indexing keywords for a set of pictures.
    """
    model1 = select_model(model_name)
    model2 = load_model(os.path.join(nn_dir, model_filename))

    # Obtain the pictures list from album_ini or pictures_dir
    inifile = find_pictures(album_ini, pictures_dir)

    for picture_name in inifile.sections():
        picture = os.path.join(pictures_dir, picture_name)
        if os.path.isfile(picture):
            try:
                keywords = picture_keywords(picture, model1, model2, lang, min_proba, nb_keywords)
                keywords = clean_keywords(keywords, lang)
                inifile[picture_name]['auto_tags'] = ','.join(keywords)
            except Exception as exception:
                logerr("Can't find keywords for picture '{}'={}. Reason: {}".format(picture, keywords, type(exception).__name__))
        else:
            if picture_name != "album":
                log("Not processing section [{}], not a picture".format(picture_name))

    return inifile


def clean_keywords(keywords, lang):
    """
    At least for the French language, NLTK WordNet French corpus is not correctly encoded and some characters
    are not correctly encoded in Python (like œ). Also, "_" is used instead of "-" in compound words.
    """
    if lang == 'fra':
        cleaned = []
        for keyword in keywords:
            keyword = keyword.replace('\x9c', 'œ')
            cleaned.append(keyword.replace('_', '-'))
        keywords = cleaned

    return keywords


def picture_keywords(filename, model1, model2, lang, min_proba, nb_keywords):
    """
    Generate the best keywords describing the picture whose filename
    is given.
    Load the image file and run it through the neural network to get the
    targets the neural network has detected. We keep either the targets with
    a prediction over `min_proba` or the given `nb_keywords`. If you want the
    most precise detection, you should use `min_proba`, but there could be
    pictures where the neural network is unable to detect items.
    Finally, it finds the keywords associated to the synsets and return them
    """
    # Get features through model1
    image = load_img(filename, target_size=TARGET_SIZE)
    image = img_to_array(image)
    # Reshape for model1
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    features = model1.predict(image)

    # Now get keywords from model2
    features = model2.predict(features)

    # As we processed only one picture, we keep only the first result
    features = features[0]

    # Find the synsets from the features
    synsets_ids = max_synsets(features, min_proba, nb_keywords)

    # Get the keywords in the requested language
    keywords = []
    for synset_id in synsets_ids:
        keyword = keyword_from_synset_id(synset_id, lang)
        if keyword is not None:
            keywords.append(keyword)

    return keywords



def max_synsets(features, min_proba, nb_keywords):
    """
    Extract the 'nb_keywords' synset ids from the given probability features.
    If `min_proba` is > 0, we use it to keep all the synsets whose probablity
    of appearance in the picture is superior to that value.
    Else we take the `nb_keywords` synsets with maximum probability of identification.
    """
    if min_proba > 0.0:
        synsets_ids = np.argwhere(features > min_proba).flatten()
    else:
        synsets_ids = np.argpartition(features, - nb_keywords)[- nb_keywords:]

    return synsets_ids


def create_album_ini(pictures):
    """
    Create a default `album.ini` file with the given pictures.
    Does not create the file on disk.
    """
    inifile = configparser.ConfigParser(allow_no_value=True)
    inifile.optionxform = str

    inifile["DEFAULT"] = {
        "# User defined metadata for MyPhotoShare": None,
        "########################################": None,
        "# Possible metadata:": None,
        "# - title: To give a title to the photo, video or album.": None,
        "# - description: A long description of the media.": None,
        "# - tags: A comma separated list of key words.": None,
        "# - date: The date the photo was taken, in the format YYYY-MM-DD.": None,
        "# - latitude: The latitude of the media, for instance if the media was not geotagged when captured.": None,
        "# - longitude: The longitude of the capture of media.": None,
        "# - country_name: The name of the country where the photo was shot.": None,
        "# - region_name: The name of the region.": None,
        "# - place_name: The name of the city or town to be displayed.": None,
        "#---------------------------------------": None,
        "#title": "",
        "#description": "",
        "#tags": "",
        "#date": "",
        "#latitude": "",
        "#longitude": "",
        "#contry_name": "",
        "#region_name": "",
        "#place_name": ""
    }

    inifile["album"] = {
        "#title": "",
        "#description": "",
        "#tags": ""
    }

    for picture in pictures:
        inifile.add_section(os.path.basename(picture))

    return inifile


# Let's go!
main()


