# check that the hosted runtime is a Tesla P100
!nvidia-smi -L

# mount google drive so that generations can be saved
from google.colab import drive
drive.mount('/content/gdrive')

# install jukebox and its necessary dependencies
!pip install git+https://github.com/openai/jukebox.git

# import necessary packages
import jukebox
import torch as t
import librosa
import os
from IPython.display import Audio
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample, \
                           load_prompts
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
rank, local_rank, device = setup_dist_from_mpi()

# sample from the 5b_lyrics model
# had to make actual song generation shorter so that the upsampling portion didn't time out
t.cuda.empty_cache() # empty cache to ensure there's as much allocated memory as possible
model = '5b_lyrics'
hps = Hyperparams()
hps.sr = 44100 # sampling rate; samples per second per channel (44100 is the default)
hps.n_samples = 1 if model in ('5b', '5b_lyrics') else 4 # changing number of samples based on available compute

# Specifies the directory to save the sample in.
# We set this to the Google Drive mount point.
hps.name = '/content/gdrive/My Drive/yeezyAI/ancestral3'
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32 # change chunk size based on available compute power
max_batch_size = 1 if model in ('5b', '5b_lyrics') else 16 # change max batch size based on available computer power
hps.levels = 3 # number of levels to upsample in final step
hps.hop_fraction = [.5,.5,.125]

vqvae, *priors = MODELS[model] # generate VQ-VAE models; creates discrete representation of waves in music (continuous)
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576)), device)
top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

# creates songs based on artist and genre conditioning
# using this instead of priming used in project 1
mode = 'ancestral'
codes_file=None
audio_file=None
prompt_length_in_seconds=None

# section of code that continues from last checkpoint of generation
# useful for when google colab times out
if os.path.exists(hps.name):
  for level in [1, 2]:
    data = f"{hps.name}/level_{level}/data.pth.tar"
    if os.path.isfile(data):
      mode = 'upsample'
      codes_file = data
      print('Upsampling from level '+str(level))
      break
print('mode is now '+mode)

sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))

sample_length_in_seconds = 40 # shorter than 60 in project 1 due to 5b_lyrics model
hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

# changed default metas from project 1 to use relevant ones (Kanye, hip hop genre)
# had to ensure the lyric length was appropriate for the given sample length, as the association is linear
metas = [dict(artist = "Kanye West",
            genre = "Hip Hop",
            total_length = hps.sample_length,
            offset = 0,
            lyrics = """
            This what the cops was sellin'
            I seen the movie, "Back to the Future"
            A gangsta outfit, with lightspeed twister
            Speedboat strap, with tinted windows
            Ventrada rosa, with cell phone chain
            Now is that your sobriety?
            Yeah that is, is
            My money's around the world, and I'm feelin' it all
            From all over the place, yeah
            I'm from the wrong place, I'm on everything
            That walrus, sweeter, drivin' it
            I might be poppin' on furs, I'm a wolf
            So I Must Scream, I'm a wolf
            I'm not suffocated by candles, I'm still feelin' it
            I'm not famvin' in bed, I'm still feelin' it
            The floors wasched, the sheets was rugs
""",
            ),
          ] * hps.n_samples
labels = [None, None, top_prior.labeller.get_batch_labels(metas, 'cuda')]

sampling_temperature = .98 # change the sampling temperature (randomness); best for jukebox to keep at .98

lower_batch_size = 16
max_batch_size = 3 if model in ('5b', '5b_lyrics') else 8
lower_level_chunk_size = 32
chunk_size = 16 if model in ('5b', '5b_lyrics') else 32
sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=lower_batch_size,
                        chunk_size=lower_level_chunk_size),
                    dict(temp=0.99, fp16=True, max_batch_size=lower_batch_size,
                         chunk_size=lower_level_chunk_size),
                    dict(temp=sampling_temperature, fp16=True, 
                         max_batch_size=max_batch_size, chunk_size=chunk_size)]

# empty cache again before big compute; will generate top level by sampling from the 1b_lyrics model and then upsample twice
t.cuda.empty_cache()
if sample_hps.mode == 'ancestral':
  zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(len(priors))]
  zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
elif sample_hps.mode == 'upsample':
  assert sample_hps.codes_file is not None
  # Load codes.
  data = t.load(sample_hps.codes_file, map_location='cpu')
  zs = [z.cuda() for z in data['zs']]
  assert zs[-1].shape[0] == hps.n_samples, f"Expected bs = {hps.n_samples}, got {zs[-1].shape[0]}"
  del data
  print('Falling through to the upsample step later in the notebook.')
elif sample_hps.mode == 'primed':
  assert sample_hps.audio_file is not None
  audio_files = sample_hps.audio_file.split(',')
  duration = (int(sample_hps.prompt_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
  x = load_prompts(audio_files, duration, hps)
  zs = top_prior.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
  zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)
else:
  raise ValueError(f'Unknown sample mode {sample_hps.mode}.')

# resultant file before upsampling (removing whitenoise)
Audio(f'{hps.name}/level_2/item_0.wav')

# upsampling
# have to delete top_prior, otherwise upsampling will run out of memory
# lyric visualization only available at top level because of this, but we simply use the lyric visualization at the top level with upsampled audio swapped in
if True:
  del top_prior
  empty_cache()
  top_prior=None
upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1]]
labels[:2] = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers]

# this step took about 8 hours
zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)

# listen to the final sample
Audio(f'{hps.name}/level_0/item_0.wav')

del upsamplers
empty_cache()

'''
The below code was used to co-sample some music, which ended up being too taxing to upsample fully,
so the final result is noisy. The result can be found in ancestral1/level_1
'''
model = "1b_lyrics" # or "1b_lyrics"
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 3 if model in ('5b', '5b_lyrics') else 16
# Specifies the directory to save the sample in.
# We set this to the Google Drive mount point.
hps.name = '/content/gdrive/My Drive/yeezyAI/ancestral1'
hps.sample_length = 1048576 if model in ('5b', '5b_lyrics') else 786432 
chunk_size = 8 if model in ('5b', '5b_lyrics') else 32
max_batch_size = 3 if model in ('5b', '5b_lyrics') else 16
hps.hop_fraction = [.5, .5, .125] 
hps.levels = 3

vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)
top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

mode = 'ancestral'
codes_file=None
audio_file=None
prompt_length_in_seconds=None

sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))

sample_length_in_seconds = 80          # Full length of musical sample to generate - we find songs in the 1 to 4 minute
                                       # range work well, with generation time proportional to sample length.  
                                       # This total length affects how quickly the model 
                                       # progresses through lyrics (model also generates differently
                                       # depending on if it thinks it's in the beginning, middle, or end of sample)
hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

metas = [dict(artist = "Kanye West",
            genre = "Hip Hop",
            total_length = hps.sample_length,
            offset = 0,
            lyrics = """
            They outside of the Saksons
            Lookin' at bank notes, notes that's saith it
            Its like they say, "You fucked up, don't laugh"
            Its like they say, "You should hangout with Mos"
            All of the time
            This what the cops was sellin'
            I seen the movie, "Back to the Future"
            A gangsta outfit, with lightspeed twister
            Speedboat strap, with tinted windows
            Ventrada rosa, with cell phone chain
            Now is that your sobriety?
            Yeah that is, is
            My money's around the world, and I'm feelin' it all
            From all over the place, yeah
            I'm from the wrong place, I'm on everything
            That walrus, sweeter, drivin' it
            I might be poppin' on furs, I'm a wolf
            So I Must Scream, I'm a wolf
            I'm not suffocated by candles, I'm still feelin' it
            I'm not famvin' in bed, I'm still feelin' it
            The floors wasched, the sheets was rugs
            Erratum, and uglier than ever
            Theophanies were made, like Michael Jackson
            Like Marie Bica's no-no, and Dior Homme's only no-no
            I'm pimply, you should call Phoebe Philo
            I'm rich, that's why another millionaire bitch
            I'm not in the house to wash my hands
            I call this house a day spa
            """,
            ),
          ] * hps.n_samples
labels = top_prior.labeller.get_batch_labels(metas, 'cuda')

def seconds_to_tokens(sec, sr, prior, chunk_size):
  tokens = sec * hps.sr // prior.raw_to_tokens
  tokens = ((tokens // chunk_size) + 1) * chunk_size
  assert tokens <= prior.n_ctx, 'Choose a shorter generation length to stay within the top prior context'
  return tokens

initial_generation_in_seconds = 4
tokens_to_sample = seconds_to_tokens(initial_generation_in_seconds, hps.sr, top_prior, chunk_size)

sampling_temperature = .98

lower_batch_size = 8
max_batch_size = 3 if model in ('5b', '5b_lyrics') else 8
lower_level_chunk_size = 32
chunk_size = 8 if model in ('5b', '5b_lyrics') else 16
sampling_kwargs = dict(temp=sampling_temperature, fp16=True, max_batch_size=lower_batch_size,
                        chunk_size=lower_level_chunk_size)

t.cuda.empty_cache()
if sample_hps.mode == 'ancestral':
  zs=[t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(3)]
  zs=sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)
elif sample_hps.mode == 'primed':
  assert sample_hps.audio_file is not None
  audio_files = sample_hps.audio_file.split(',')
  duration = (int(sample_hps.prompt_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
  x = load_prompts(audio_files, duration, hps)
  zs = top_prior.encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
  zs = sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)
x = vqvae.decode(zs[2:], start_level=2).cpu().numpy()

for i in range(hps.n_samples):
  librosa.output.write_wav(f'noisy_top_level_generation_{i}.wav', x[i], sr=44100)

Audio('noisy_top_level_generation_0.wav')

Audio('noisy_top_level_generation_1.wav')

Audio('noisy_top_level_generation_2.wav')

my_choice=1

zs[2]=zs[2][my_choice].repeat(hps.n_samples,1)
t.save(zs, 'zs-checkpoint2.t')

# Set to True to load the previous checkpoint:
if True:
  zs=t.load('zs-checkpoint2.t')

continue_generation_in_seconds=4
tokens_to_sample = seconds_to_tokens(continue_generation_in_seconds, hps.sr, top_prior, chunk_size)

zs = sample_partial_window(zs, labels, sampling_kwargs, 2, top_prior, tokens_to_sample, hps)
x = vqvae.decode(zs[2:], start_level=2).cpu().numpy()

playback_start_time_in_seconds = 34

for i in range(hps.n_samples):
  librosa.output.write_wav(f'top_level_continuation_{i}.wav', x[i][playback_start_time_in_seconds*44100:], sr=44100)

Audio('top_level_continuation_0.wav')

Audio('top_level_continuation_1.wav')

Audio('top_level_continuation_2.wav')

choice = 2
select_best_sample = True  # Set false if you want to upsample all your samples 
                           # upsampling sometimes yields subtly different results on multiple runs,
                           # so this way you can choose your favorite upsampling

if select_best_sample:
  zs[2]=zs[2][choice].repeat(zs[2].shape[0],1)

t.save(zs, 'zs-top-level-final.t')

model = "1b_lyrics" # or "1b_lyrics"
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 3 if model in ('5b', '5b_lyrics') else 16
hps.name = '/content/gdrive/My Drive/yeezyAI/ancestral1'
hps.sample_length = 1048576 if model in ('5b', '5b_lyrics') else 786432 
chunk_size = 8 if model in ('5b', '5b_lyrics') else 32
max_batch_size = 3 if model in ('5b', '5b_lyrics') else 16
hps.hop_fraction = [.5, .5, .125] 
hps.levels = 3

if True:
  zs = t.load('zs-top-level-final.t')

assert zs[2].shape[1]>=2048, f'Please first generate at least 2048 tokens at the top level, currently you have {zs[2].shape[1]}'
hps.sample_length = zs[2].shape[1]*top_prior.raw_to_tokens

if True:
  del top_prior
  empty_cache()
  top_prior=None

print(priors)

upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1]]

sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=16, chunk_size=32),
                    dict(temp=0.99, fp16=True, max_batch_size=16, chunk_size=32),
                    None]

if type(labels)==dict:
  labels = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers] + [labels]

zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)

Audio(f'{hps.name}/level_0/item_0.wav')