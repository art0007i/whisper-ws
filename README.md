# whisper-ws

## Usage

### 1. Compile whisper.cpp
```bash
cd whisper_gpu
cmake -B build -S .
cmake --build build -j
```

### 2. Download whisper model
```bash
./models/download-ggml-model.cmd base
```
instead of 'base' you can download other models which can be better or faster or not I haven't tried them
- tiny
- base
- small
- medium
- large-v1
- large-v2
- large-v3
- large-v3-turbo

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

### 6.. Run the script
```bash
python main.py
```

### 7. Connect in Resonite: `ws://127.0.0.1:6912/ws`
It will send messages like
```
lang=en
start=0.1
end=1.1
text=Hello
```
