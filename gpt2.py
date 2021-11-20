import gpt_2_simple as gpt2
import os
import requests

# Small-sized model, only one that doesn't run out of memory before completion
model_name = "124M"

# Downloads model from GPT-2 if not existent in models directory
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    # model is saved into current directory under /models/124M/
    gpt2.download_gpt2(model_name=model_name)

# Data set of all kanye verses from his career up till 2018
kanye_verses = "kanye_verses.txt"
# Data set of all kanye verses 2018 - present
kanye_post2018 = "kanye_post2018.txt"

# Run this once for kanye_verse.txt, and once for kanye_post2018.txt
session1 = gpt2.start_tf_sess()
gpt2.finetune(session1,
              kanye_verses,
              model_name=model_name,
              steps=1000,  # steps is max number of training steps
              restore_from='fresh',  # doing stacked generation didn't work that well
              run_name='run1',
              print_every=20,  # Print epoch progress every 20 steps
              sample_every=200,  # Show sampled data every 200 steps
              save_every=500  # Save data to checkpoint every 500 steps
              )

gpt2.generate(session1)
