# filler-sound-detector
This project aims at detecting annoying filler sounds in an audio with help of ML. 

# Data
I labeled the data myself while editing my friend/client's podcast as a side hustle. There are a total of 67 audio tracks (22.3 GB) and text files with labels. The labels show start and end timestamps of the audio with filler sounds.
- Language: Spanish
- Format: Waveform audio
- Sampling rate: 48k, 16 bit

Additional publicly available data may be used as well as augmentation techniques (adding background noise and similar).

# Stragegy

While preprocessing, we split the data into equally long chunks. Each chunk is mapped to not-filler/filler. Possibly, a hopping window can be used.

![](img/stragegy-1.png)

To decide what is a filler, we will look at what amount of time of a sound is filler and if it's above a threshold (50%/70%/90%), it will be labeled as such.

Window and hopping length can be adjusted after evaluation to find the optimal tradeoff between processing speed and performance.

# Exploratory data analysis

### Labels

To choose an appropriate window and hop length for our data preprocessing, we have to consider the target distribution.

The filler sound length seems to follow a log-normal distribution. 

![](img/target_distribution.jpg)

Turns out in 95% of fillers last longer than 0.279 sec. A good fit for window length, therefore, would be anything below `sample_rate * 0.279`. Given that we down-sample the audio from 48k samples/sec to 8k, we get `2232` samples per window. We'll round that number to `2048`.
