# whisper-ws

lets you use speech to text in resonite (or anywhere else over a websocket)

btw this is like 90% vibecoded. but the readme is at least 100% human

I only tested it on linux. I hope it works on windows too but it probably does not :)

## Usage

### 0. Clone the repo
```bash
git clone https://github.com/art0007i/whisper-ws.git
cd whisper-ws
```

### 1. Compile whisper.cpp
```bash
cd whisper_gpu
cmake -B build -S .
cmake --build build -j
```

### 2. Download whisper model
```bash
./models/download-ggml-model.cmd large-v3-turbo
```
instead of 'large-v3-turbo' you can download other models from this list:
- tiny
- base
- small
- medium
- large-v1
- large-v2
- large-v3
- large-v3-turbo

if you choose a model different than 'large-v3-turbo' you will need to adjust main.py on the line that has `WHISPER_MODEL = `

### 3. Create venv
```bash
python -m venv .venv
```

### 4. Enter venv
https://docs.python.org/3/library/venv.html#how-venvs-work

### 5. Install requirements
```bash
pip install -r requirements.txt
```

### 5.1. Test whisper model
```bash
python gputest.py
```

### 6.. Run the script
```bash
python main.py
```

### 7. Connect in Resonite: `ws://127.0.0.1:6912/ws`
It will send messages like
```
type=final
lang=en
start=0.1
end=1.1
text=Hello
```

Explanation
- type
    - final: gets sent when you stop talking. text seems accurate
    - interim: gets sent in realtime, during talking. usually garbage but can be useful
- lang: the language that whisper thinks you are speaking
- start: the time that this segment began in
- end: the time that this segment ended in
- text: obvious 

crappy client implementation:
```
resdb:///fc4430bd4e5b234429ec24def142962167944f0dfded3382620a89b8509b2a43.brson
```
