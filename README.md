# mps_autotags

Generate automatic keywords index from the picture files in a MyPhotoShare album using deep learning.
It uses a neural network to recognize scenes into pictures and generate keywords from their content.
Keywords are generated in the language specified.

* Author: Pierre Métras
* Date: 2020-04-01 April 1st fools release
* Version: 1.0

## Notes

* Google Tensorflow is very picky on versions of software. The recipe for install below has been tested
on Ubuntu 18.04 LTS and Arch Linux, 64 bits. Tensorflow does not work on 32 bits systems. Using a virtual
environment makes it reproducible.

* You will need a lot of disk space and memory to try this project. 100 GB free space on disk or SSD with
32 GB of RAM are required for some scenarios. The default model can run with 70 GB disk space and 16 GB RAM.

## Install

We use Python 3 virtual environment

* Tensorflow including Keras

```bash
$ sudo apt install python3-venv

# Create Python 3 virtual environment 'venv'
$ python3 -m venv venv

# Activate venv
$ source venv/bin/activate

# Upgrade pip
(venv) $ pip install --upgrade pip

# Install Tensorflow
(venv) $ pip install --upgrade tensorflow
```

* NLTK

```bash
(venv) $ pip install --upgrade nltk

$ python3
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('omw')
>>> nltk.download('stopwords')
>>> exit()
```

* Pillow

```bash
(venv) $ pip install --upgrade Pillow
```

* psutil

```bash
(venv) $ pip install --upgrade psutil
```

* sklearn

```bash
(venv) $ $ pip install --upgrade sklearn
```

## Data files

The source data files must be downloaded from external web sites.

### Visual Genome files

Web site: https://visualgenome.org/

Download the following files from the page [Visual Genome Download](http://visualgenome.org/api/v0/api_home.html)

#### data/vgenome

* [Objects](http://visualgenome.org/static/data/dataset/objects.json.zip)
* [Relationships](http://visualgenome.org/static/data/dataset/relationships.json.zip)
* [Synsets for objects](http://visualgenome.org/static/data/dataset/object_synsets.json.zip)
* [Image metadata](http://visualgenome.org/static/data/dataset/image_data.json.zip)
* [Synsets](http://visualgenome.org/static/data/dataset/synsets.json.zip)

#### data/vgenome/images

Unzip these 2 files into the directory `data/vgenome/images`.

* [https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip)
* [https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

### MS COCO files

Web site: https://cocodataset.org/

Download the following files from the page [COCO Dataset Download](http://cocodataset.org/#download)

#### data/mscoco

* [Annotations Train Val 2014](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

#### data/mscoco/images

* [Train images 2014](http://images.cocodataset.org/zips/train2014.zip)
* [Val images 2014](http://images.cocodataset.org/zips/val2014.zip)
* [Test images 2014](http://images.cocodataset.org/zips/test2014.zip)

## Running mps_autotags

Running mps_autotags is a 3 steps process:

1. Prepare the data with `mps_autotags init`.
2. Train the neural network with `mps_autotags learn`.
3. Tag MyPhotoShare pictures with `mps_autotags generate`.

The first two steps are CPU intensive and usually will take multiple days to complete.

### mps_autotags init

This step analyzes the data and create work files. You select the type of neural network to use for scene recognition in
pictures and you pre-process the training pictures for this network.

```echo
usage: mps_autotags init [--options]

Create environment from Visual Genome and MS COCO files

optional arguments:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Directory where mps_autotags saves its data files
                        (default=data/data)
  --vgenome-dir VGENOME_DIR
                        Directory with the training image data from Visual
                        Genome (default=data/vgenome)
  --mscoco-dir MSCOCO_DIR
                        Directory with the training image data from MS COCO
                        (default=data/mscoco)
  --model MODEL         Pretrained neural network model architecture
                        (default=xception)
  --augment AUGMENT     Data augmentation: number of augmented image variants
                        created (default=1)
  --no-features         Don't extract features from images
  --add-attributes      Add keywords from Visual Genome objects attributes
  --add-relations       Add keywords from Visual Genome objects relationships
  --add-synsets         Add synsets found in MS COCO captions even if they
                        don't exist in Visual Genome
  --limit-pictures LIMIT_PICTURES
                        Limit the number of pictures to extract features from,
                        useful for debugging
```

Prepare mps_autotags environment to train a neural network to recognize features from pictures.
Pictures dataset are taken from Visual Genome and MS COCO datasets.
Keywords are learnt from Visual Genome Wordnet picture descriptions and MS COCO captions.

For this first version of mps_autotags, only two neural networks are implemented, embedded with
a 2 layers multi-targets classification output.

* xception: [Xception: Deep Learning with Depthwise Separable Convolutions]()
* vgg16: [VGG16: Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

Other image recognition neural network architectures can easily be added to the code. Look at `select_model(model_name)`.

One can generate artificial pictures from the original datasets with the `--augment=N` option, giving more
training data to the neural network for better results but also slowing the training process by a factor of N.
Picture augmentation consists of:

* Rotation up to 15 degrees
* Heigh shift up to 10%
* Width shift up to 10%
* Horizontal flip
* Zoom in the range [90%, 120%]

### mps_autotags learn

This step trains the classification network to associate picture scenes with keywords. When this step completes, the best
trained neural network will be save to be used in picture recognition.

```echo
usage: mps_autotags learn [--options]

optional arguments:
  -h, --help            show this help message and exit
  --max-epochs MAX_EPOCHS
                        Maximum number of epochs (default=30)
  --batch-size BATCH_SIZE
                        Size of training batches (default=32)
  --val-percent VAL_PERCENT
                        Percentage of training images used for validation
                        (default=10%)
```

Train a neural network to associate caption keywords with image features that were prepared during the 'init' phase.
The neural network will be trained for a maximum of `--max-epochs` epochs with gradient descent.
`--val-percent` percents of the pictures will be used to validate and guide the training process.

### mps_autotags generate

You run the `generate` command to recognize scenes from MyPhotoShare picture albums. Metadata files are generated or
updated with the keywords recognized in the pictures, in the specified language.

```echo
usage: mps_autotags generate --lang=<ISO-639-3> [--options]

optional arguments:
  -h, --help            show this help message and exit
  --lang LANG           Language used to generate keywords (default=eng)
  --nb-keywords NB_KEYWORDS
                        Number of keywords to generate (default=5)
  --min-proba MIN_PROBA
                        Minimal probability to keep a keyword
  --album-dir ALBUM_DIR
                        Directory containing the pictures that will be indexed
  --album-ini ALBUM_INI
                        Path to the 'album.ini' file containing the pictures
                        that will be indexed
  --no-update           Don't write generated keywords into `album.ini` file
```

Create or update exiting `album.ini` file in the specified location with automatic keywords
for each picture in the `album.ini` file or in the directory.

Either `--album-dir` or `--album-ini` must be specified. If `album-dir` is given and there
exists an `album.ini` file, the content of that file will be used to find the pictures instead
of the content of the album directory.

To prevent overwriting `album.ini` file, use `--no-update`.

By default, 5 keywords are generated for each picture. If you want to keep only pertinent
keywords, you can use `--min-proba` where the software will only keep keywords with a confidence
probability over the value given. It can happen that no keywords are generated.

Note that `--lang` code is a ISO-639 3 characters language code like:
    ['als', 'arb', 'bul', 'cat', 'cmn', 'dan', 'ell', 'eng', 'eus', 'fas', 'fin',
     'fra', 'glg', 'heb', 'hrv', 'ind', 'ita', 'jpn', 'nld', 'nno', 'nob', 'pol',
     'por', 'qcn', 'slv', 'spa', 'swe', 'tha', 'zsm']

## Config file

The 'init' phase options and results are saved in the `data/data/config.ini` file. This files can be
hand-edited to add calculation results from another computer for instance or to do tests.

### Config file example

```ini
[DEFAULT]
data_dir = ... edited .../mps_autotags/data/data
vgenome_dir = ... edited .../mps_autotags/data/vgenome
mscoco_dir = ... edited .../mps_autotags/data/mscoco
nn_dir = ... edited ...//mps_autotags/data/nn
current_model = xception
nb_synsets = 20383

[xception]
augment = 10
limit_pictures = -1
model_filename = xception.h5
max_epochs = 30
val_percent = 10

#[xception]
#augment = 1
#limit_pictures = -1
#model_filename = xception.h5
#max_epochs = 30
#val_percent = 10

[vgg16]
augment = 1
limit_pictures = -1
model_filename = vgg16.h5
max_epochs = 30
val_percent = 10
```

## Calculation time

A neural network take a lot of time and computing resources to learn, compared with a child! Be
prepared to wait for the computer to produce results...

The `init` phase will take from a few hours to one day, depending on the number of pictures you
want to prepare and the type of neural network selected.

The `learn` phase is the real long one! With a large pictures set, it will take a few days to train
the neural network. Here again, it depends on the number of pictures. The more, the longer. For instance,
on a PC with an AMD Rizen 3 with 12 cores and 32 GB or memory, it took 2 weeks to train a 1-million
pictures dataset.

I tried to setup an old computer with an Nvidia GPU to use CUDA with tensorflow, but the CPU was
too old and tensorflow did not support some vectorized instructions. Probably that GPU-based
tensorflow would reduce the calculation time. If you have access to cloud resources, you could
have a try at it.

Depending on the quantity of memory available, calculations will be done entirely in memory or
with batching. Even with 100k prepared pictures, the 32 GB memory I had access to was not enough
to do memory-only calculations. I've only been able to test it on reduced datasets.

The `generate` phase is not optimized but is the fastest of all, particularly as it runs only on one
directory. You get results in a few seconds.

## Results

Well, not that good at the present time. See the [Future work](#future-work) section to find how to improve these
results.

### Examples

* Generate 5 keywords for all pictures in the Test directory

The name of the pictures, in French, should give you indications on the content of the pictures... Sorry, these are
personal private pictures and I can't share them.

```echo
$ ./mps_autotags generate --lang=eng --album-dir=Test --no-update
2 - Generate
2 - Load synsets list
2 - Loaded 20383 synsets
2 - Selecting model from parameters
2020-03-30 20:03:50.333982: E tensorflow/stream_executor/cuda/cuda_driver.cc:351] failed call to cuInit: UNKNOWN ERROR (303)
== Generated Keywords ===
Picture filename --> Tags
-------------------------
Bernaches.jpg --> person,sky,airplane,kite,fly
Orchidée.jpg --> field,flower,tree,green,clear
La grande verrière de la Gare de l'Est.jpg --> tall,walk,side,floor,ceiling
Un bouton de pavot avant la floraison.jpg --> sweater,tree,flower,face,together
Le vieux pommier de St-André.jpg --> cloudy,field,sky,tree,branch
Un petit goûter avant l'effort.jpg --> home_plate,kitchen,flower,fork,table
TGV à l'arrêt.jpg --> light,tall,bridge,water,London
Marie-Claude.jpg --> top,floor,little,woman,bathroom
Pavot.jpg --> plant,tree,petal,vase,flower
La gare de Romilly-sur-Seine.jpg --> front,building,train,path,brick
Lac.jpg --> body,boat,river,water,lake
-------------------------
```

* Generate keywords with a probability of detection larger than 0.8

You'll notice that the neural network did not recognized items in some pictures.

```echo
$ ./mps_autotags generate --lang=eng --album-dir=Test --no-update --min-proba=0.8
2 - Generate
2 - Load synsets list
2 - Loaded 20383 synsets
2 - Selecting model from parameters
2020-03-30 20:11:10.553373: E tensorflow/stream_executor/cuda/cuda_driver.cc:351] failed call to cuInit: UNKNOWN ERROR (303)
== Generated Keywords ===
Picture filename --> Tags
-------------------------
Bernaches.jpg --> airplane,fly,kite
Orchidée.jpg -->
La grande verrière de la Gare de l'Est.jpg -->
Première étape devant Notre-Dame de Paris.jpg --> clock
Le vieux pommier de St-André.jpg --> tree,sky,branch
Un petit goûter avant l'effort.jpg --> table
TGV à l'arrêt.jpg --> water,bridge,London
Marie-Claude.jpg -->
Pavot.jpg --> flower,vase,petal
Axelle en pleine jasette avec un cerf de Virginie.jpg -->
La gare de Romilly-sur-Seine.jpg --> train
Lac.jpg --> water,boat,river,lake
-------------------------
```

## Logging

The code prints logging information on the console to show that it is still running. The number at the start
of the line is the number of CPU-seconds the program has been running.

## Tracking training

As this code is using [Tensorflow-based keras API](https://keras.io/), training can be followed with
[Tensorboard](https://www.tensorflow.org/tensorboard). During the long hours you mondel is training, open a console
and run:

```sh
$ tensorboard --logdir=data/nn
2020-03-30 20:40:05.713065: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
2020-03-30 20:40:05.713170: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
2020-03-30 20:40:05.713192: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.1.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

Then point your web browser at <http://localhost:6006/> to look at the network architecture and it's learning curve...

## Warnings

* Tensorflow prints some warnings when it can't load CUDA libraries. Informational messages and regular
warning have been disabled as they polute the program output but you can enable them.

## Bugs

* See the [Results](#results) section. Could we say that a neural network produces bugs when it does not produce the
expected results? I'll let you decide.

* Some pictures are not found in the Python `images` dictionary but I haven't found the courage to debug
it. One or two missing pictures over >100k images is something I can live with for the moment...

* WordNet terms for French language are of poor quality. There are problems with encoding of characters, wrong translations
from English, etc.

## Future work

Time is money in the deep learning race and my computers are too old to win in that race. The first thing would be
to use a modern computer with lot of CPU and GPU power and lot of memory. Or to be more eco-friendly and pay for
usage of a large cloud instance with TPU or GPU units. This would reduce the delay between trying a new hypothesis
and looking at results.

This code was started because I wanted an international keywords generator to index my photos collection. Most examples
on the Web use English captions by MyPhotoShare is used by non-English speakers. Visual Genome pictures dataset seemed a good dataset to use because pictures where described using WordNet synsets, and there are multiple localized WordNet available.
Unfortunately, the dataset seems too small (~108k pictures) for 18k synsets vocabulary. So either increase the pictures
dataset or reduce the vocabulary...

There are now larger image datasets like [MS COCO](http://cocodataset.org/#home) or
[Google Open Image Dataset](https://opensource.google/projects/open-images-dataset) that contain millions of
annotated pictures. Developing the natural language part of the program would allow to translate English
descriptions to WordNet synsets and then to process a larget set of pictures, hopefully
bringing better results. But I don't have the computing power to go that path for the moment...
