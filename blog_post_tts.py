#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TTS.api import TTS
tts = TTS(model_name="tts_models/en/vctk/vits", gpu=True)
sample_rate = tts.synthesizer.output_sample_rate


# In[2]:


import requests
from bs4 import BeautifulSoup
import sys
from urllib.parse import urlparse


# In[12]:


if 'ipykernel' in sys.modules:
    url = "https://www.elbeno.com/blog/?p=1725"
else:
    url = sys.argv[1]

parsed_url = urlparse(url)
path = parsed_url.path

# Split the path by slashes and take the last element
if path[-1] == '/':
    path = path[:-1]

output_file_stem = path.split('/')[-1]

# Fetch the HTML content
response = requests.get(url)
html_content = response.text
print(html_content)


# In[5]:


# Parse the HTML content with Beautiful Soup
soup = BeautifulSoup(html_content, 'html.parser')

# Find and remove all <code> elements
for code_tag in soup.find_all('pre'):
    code_tag.decompose()

# Extract text and remove leading/trailing whitespace
text_content = soup.get_text(separator='\n', strip=True)
print(text_content)


# In[6]:


from nltk.data import load


# In[7]:


tokenizer = load(f"tokenizers/punkt/english.pickle")
sentences = tokenizer.tokenize(text_content)
print(sentences)


# In[13]:


import subprocess
from TTS.tts.utils.synthesis import synthesis

command = [
            'ffmpeg', '-y',
            '-f', 'f32le', 
            '-ar', str(sample_rate),  # sample rate
            '-ac', '1',  # number of audio channels
            '-i', '-',  # The input comes from stdin
            '-acodec', 'copy',  # audio codec for M4A
            output_file_stem + '.wav',
        ]
print(" ".join(command))

process = subprocess.Popen(command, stdin=subprocess.PIPE)

for sentence in sentences:
    wav = synthesis(
            model=tts.synthesizer.tts_model,
            text=sentence,
            CONFIG=tts.synthesizer.tts_config,
            use_cuda=True,
            speaker_id=77, # p307
            use_griffin_lim=tts.synthesizer.vocoder_model is None,
            )["wav"]
    format = wav
    process.stdin.write(wav.tobytes())

process.stdin.close()
process.wait()
print("------------\n", format)


# In[19]:


print(sys.argv)


# In[ ]:




