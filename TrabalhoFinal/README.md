# Dashboard - Saúde Mental no Brasil

Este repositório contém um dashboard interativo construído com Plotly Dash que apresenta uma análise da série "Saúde Mental em Dados" (período 2002–2024). O aplicativo principal é `dashboard_saude_mental.py`.

## Requisitos
- Python 3.8+ (o projeto foi testado em ambientes Python 3.10 - 3.13; use a sua versão preferida do Python 3)
- Virtualenv (recomendado)
- Arquivo de dependências: `requirements.txt`

## Estrutura principal
- `dashboard_saude_mental.py` — script principal que inicia o dashboard
- `requirements.txt` — dependências do projeto

## Instruções de execução

As instruções abaixo mostram como criar um ambiente virtual, instalar dependências e executar o dashboard em macOS (zsh / bash) e Windows (PowerShell / CMD).

OBS: em sistemas macOS/Linux, prefira `python3` se `python` apontar para Python 2.x.

### macOS (zsh / bash)

1. Abra o Terminal e navegue até a pasta do projeto.

2. Crie um ambiente virtual (opcional, recomendado):

```bash
python3 -m venv .venv
```

3. Ative o ambiente virtual:

```bash
source .venv/bin/activate
```

4. Instale as dependências:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. Execute o dashboard:

```bash
python3 dashboard_saude_mental.py
```

6. Acesse no navegador: http://127.0.0.1:8050

Para interromper o servidor: Ctrl+C no terminal.

### Windows (PowerShell)

1. Abra o PowerShell e navegue até a pasta do projeto.

2. Crie o ambiente virtual:

```powershell
python -m venv .venv
```

3. Ative o ambiente virtual (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Se receber um erro de política de execução (ExecutionPolicy), execute o PowerShell como administrador e permita scripts ou use:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

4. Instale as dependências:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

5. Execute o dashboard:

```powershell
python dashboard_saude_mental.py
```

6. Abra no navegador: http://127.0.0.1:8050

### Windows (Prompt de Comando - CMD)

1. Crie e ative venv:

```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

2. Instale dependências e execute:

```cmd
pip install -r requirements.txt
python dashboard_saude_mental.py
```


Autores: Gabriel Loures e Kauã Fernandes - Projeto: Trabalho Final - Modelagem e Programação Estatística