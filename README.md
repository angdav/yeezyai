# yeezyAI

A project aimed at creating a Kanye-themed song purely from AI and machine learning, using the power of Python, OpenAI GPT-2, Tensorflow, and OpenAI Jukebox. Ultimate goal is to make a song that could plausibly have been written, produced, and performed by Kanye West.

## Process

### Lyric Generation

- Initially used basic NLP principles like tokenizing words to numbers/sentiments, and a Bidirectional Long Short-Term Memory model to generate text based on a Kanye Verse Kaggle sample (shown in nlp.py and nlp2.py)
  - Found that those results were largely unintelligible; I would have to cherry pick results and even then it was mostly gibberish and a lot of repeating words (shown in nlp-output.txt and nlp2-output.txt)
  - They also have limitations in that they can only generate text using the words provided
- Decided to use GPT-2 models instead of my manually trained NLP models, and was able to get much better results (shown in gpt2.py and gpt2-verses-output.txt)
  - GPT-2 already has a model 40GB of English, but uses a given dataset (Kanye lyrics) to steer it a certain direction; this makes it use words outside of the given dataset
  - Since the Kaggle dataset was largely from Kanye's verses from pre-2017 (very vulgar and mostly stereotypical rap content), I decided to construct my own dataset from Kanye's more recent work ("family-approved")
    - Put this output into gpt2-post2018-output.txt
    - Tried using checkpoints with the trained data (using the generated data to train more), but at that point the sample and subject matter was too small; some parts were straight copies from actual lyrics, and most resembled it too closely
  - So, I stick with the initial generation from the larger dataset to use for music generation

### Music Generation

- Premise is to use:
  - lyrics generated from kanye_verses.txt (entire discography until 2018) -> plug it into OpenAI Jukebox
  - proper metadata
  - co-sampling to guide direction of the song as it's generated
  - 5b_lyrics model instead of 1b_lyrics
  - longer priming sample
  - repeated generation using windowed sampling to create a longer song
  - different styles (rock, pop, etc.) when providing metadata
