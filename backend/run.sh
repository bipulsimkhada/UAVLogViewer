export OPENAI_API_KEY="your_openai_api_key_here"
# export DEBUG="true" # Uncomment to enable debug mode
#!/bin/bash

set -e

# Detect platform
OS=$(uname -s)

# Define venv paths for Linux/macOS vs Windows
if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then
  VENV_DIR=".venv"
  ACTIVATE_SCRIPT=".venv/bin/activate"
else
  VENV_DIR=".venv"
  ACTIVATE_SCRIPT=".venv/Scripts/activate"
fi

# 1. Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python -m venv "$VENV_DIR"
fi

# 2. Activate the venv
if [ -f "$ACTIVATE_SCRIPT" ]; then
  echo "Activating virtual environment..."
  source "$ACTIVATE_SCRIPT"
else
  echo "❌ Cannot find activate script at $ACTIVATE_SCRIPT"
  exit 1
fi

# 3. Compile dependencies using uv and install
echo "Compiling dependencies from pyproject.toml..."
uv pip compile pyproject.toml -o requirements.txt

echo "Installing dependencies..."
uv pip install -r requirements.txt

# 4. Run the FastAPI app
echo "Running FastAPI app..."
python -m uvicorn main:app --timeout-keep-alive 90 --reload