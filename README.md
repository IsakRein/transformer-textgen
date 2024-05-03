# dd2424-project

### Hur man använder venv och pip typ:
Skapa environment i din mapp:
```sh
python -m venv .
```

Aktivera environmentet:
```sh
source ./bin/activate
```

Installera requirementsen:
```sh
pip install -r requirements.txt
```

Installera pytorch-cpu:
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Om man har nya requirements som man vill dela med andra:
```sh
pip freeze > requirements.txt
```

