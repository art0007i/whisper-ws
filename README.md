# whisper-ws

## Usage

1. Create venv
```bash
python -m venv .venv
```

2. Enter venv

https://docs.python.org/3/library/venv.html#how-venvs-work

3. Install requirements
```bash
pip install -r requirements.txt
```

4. Run the script
```bash
python main.py
```

5. Connect in Resonite: `ws://127.0.0.1:6912/ws`
It will send messages like
```
lang=en
start=0.1
end=1.1
text=Hello
```
