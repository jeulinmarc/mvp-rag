# Installation & configuration

Guide d'installation pas-à-pas et configuration des fournisseurs LLM.
Pour l'usage courant, reviens au [README](../README.md).

## Installation

### Prérequis

- **macOS** (testé sur M3 16 Go) ou Linux. Windows possible via WSL2 mais non testé.
- **8 Go RAM minimum**, 16 Go recommandés (pour faire tourner un LLM local de 7B paramètres confortablement).
- **5 Go d'espace disque** (Docker image Qdrant + modèle d'embedding + modèle LLM Ollama).

### 1. Cloner le repo

```bash
git clone https://github.com/jeulinmarc/mvp-rag.git
cd mvp-rag
```

### 2. Installer Docker Desktop

Docker fait tourner Qdrant dans un conteneur isolé. Sans Docker, il faudrait installer Qdrant en natif (compliqué).

**macOS** : télécharge Docker Desktop sur [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/). Choisis la version qui correspond à ton chip :

- **Apple Silicon** (M1/M2/M3/M4)
- **Intel**

Installe le `.dmg`, lance Docker Desktop une fois pour qu'il finalise le setup.

Alternatives plus légères si Docker Desktop te paraît lourd : [OrbStack](https://orbstack.dev), Colima, Rancher Desktop.

**Linux** : suis le guide officiel [docs.docker.com/engine/install](https://docs.docker.com/engine/install/).

**Vérifie** :

```bash
docker --version
docker compose version
```

Les deux commandes doivent répondre avec un numéro de version.

### 3. Installer Python 3.10+

**macOS** (via Homebrew, recommandé) :

```bash
brew install python@3.12
```

**Vérifie** :

```bash
python3 --version
```

### 4. Installer Ollama (LLM local)

Ollama fait tourner des LLMs open-source en local. Aucun compte, aucune clé API.

**macOS** :

```bash
brew install ollama
```

**Linux** :

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Lance Ollama en service système** (démarre automatiquement, survit au reboot) :

```bash
brew services start ollama          # macOS
# ou
sudo systemctl enable --now ollama   # Linux
```

**Télécharge le modèle par défaut** (~4.7 Go, prends ~5 min selon ta connexion) :

```bash
ollama pull qwen2.5:7b
```

**Vérifie** :

```bash
curl http://localhost:11434/api/tags
```

Doit retourner un JSON listant les modèles installés.

### 5. Setup Python et dépendances

Le venv vit à la **racine du repo** (`mvp-rag/venv/`), pas dans `src/`. Depuis la racine `mvp-rag/` :

```bash
# Créer le venv (à la racine du repo)
python3 -m venv venv

# Activer le venv (à refaire à chaque session)
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

Tu sauras que le venv est actif quand ton prompt commence par `(venv)`. L'activation est indépendante du dossier courant : tu lances **Qdrant depuis la racine** (`docker compose up -d`) et **Streamlit depuis `src/`** (`cd src && streamlit run streamlit_app.py`).

### 6. Configurer les variables d'environnement

Copie le gabarit fourni, à la **racine du repo**, puis ajuste si besoin :

```bash
cp .env.example .env
```

Contenu minimal :

```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
LLM_PROVIDER=ollama
```

Le `.env` est dans le `.gitignore` (jamais commité) ; seul `.env.example` est versionné.

### 7. Lancer Qdrant

Depuis la **racine du repo**, avec Docker Desktop actif :

```bash
docker compose up -d
```

Le `-d` (detached) lance le conteneur en arrière-plan. Vérifie :

```bash
curl http://localhost:6333/
```

Doit répondre par un petit JSON contenant `"title":"qdrant"`. Tu peux aussi ouvrir [http://localhost:6333/dashboard](http://localhost:6333/dashboard) dans ton navigateur — c'est l'UI web de Qdrant.

---

---

## Changer de LLM

Le code supporte n'importe quelle API OpenAI-compatible. Édite `.env` :

### Ollama (défaut, gratuit, local)

```
LLM_PROVIDER=ollama
```

Modèles supportés out-of-the-box : `qwen2.5:7b`, `qwen2.5:3b`, `llama3.1:8b`, etc. Change `PROVIDER_CONFIG["ollama"]["model"]` dans `ask_llm.py`.

### Nebius (cloud, payant après crédits gratuits)

```
LLM_PROVIDER=nebius
NEBIUS_API_KEY=ton_token
```

Compte sur [studio.nebius.com](https://studio.nebius.com) (carte bancaire requise depuis fin 2025).

### Groq (cloud, gratuit, sans CB)

Ajoute Groq dans `PROVIDER_CONFIG` de `ask_llm.py` :

```python
"groq": {
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": os.getenv("GROQ_API_KEY", ""),
    "model": "llama-3.3-70b-versatile",
},
```

Compte sur [console.groq.com](https://console.groq.com), puis dans `.env` :

```
LLM_PROVIDER=groq
GROQ_API_KEY=ton_token
```

---
