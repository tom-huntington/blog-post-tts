#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import sys
from urllib.parse import urlparse


# In[10]:


if 'ipykernel' in sys.modules:
    sys.argv = [
        sys.argv[0],
        "/mnt/c/Users/tynda/Downloads/Russell, James C - The Germanization of early medieval Christianity _ a sociohistorical approach to religious transformation-Oxford University Press (1994).epub",
    ]

pathOrUrl = sys.argv[1]

if pathOrUrl.startswith('http'):
    parsed_url = urlparse(pathOrUrl)
    path = parsed_url.path

    # Split the path by slashes and take the last element
    if path[-1] == '/':
        path = path[:-1]

    output_file_stem = path.split('/')[-1]

    # Fetch the HTML content
    import requests
    response = requests.get(pathOrUrl)
    html_content = response.text
else:
    from pathlib import Path
    path = Path(pathOrUrl)
    output_file_stem = path.stem
    with open(pathOrUrl) as f:
        print(sys.version)
        print(f'{path.suffix=}')
        if path.suffix == '.epub':
            from ebooklib import epub
            import ebooklib
            book = epub.read_epub(pathOrUrl)
            text = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text.append(soup.get_text())
        
            html_content = '\n'.join(text)
        else:
            html_content = f.read()


# In[11]:


# Parse the HTML content with Beautiful Soup
soup = BeautifulSoup(html_content, 'html.parser')

print(f"{len(sys.argv)=}")
if len(sys.argv) == 2:
    article = soup
else:
    article = soup.find(sys.argv[2]) if len(sys.argv) < 4 else soup.find(sys.argv[2], class_=sys.argv[3])
# article = soup.find(sys.argv[2], class_=sys.argv[3])

if not article:
    print("no atricle element")
    body = soup.body
    # for header in body.find_all('script'):
    #     header.decompose()
    
    # for script in body.find_all('script'):
    #     script.decompose()

    # for svg in body.find_all('svg'):
    #     svg.decompose()

    print(body)
    print(sys.argv[1:])
    exit(1)

# Find and remove all <code> elements
for code_tag in article.find_all(['pre', 'math', 'code']):
    code_tag.decompose()

# Extract text and remove leading/trailing whitespace
text_content = article.get_text(separator='\n', strip=True)
print(text_content)


# In[12]:


from nltk.data import load
from TTS.api import TTS
tts = TTS(model_name="tts_models/en/vctk/vits", gpu=True)
sample_rate = tts.synthesizer.output_sample_rate


# In[13]:


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




