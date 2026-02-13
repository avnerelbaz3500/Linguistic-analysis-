#!/bin/bash
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "jupyter.defaultKernel": "python"
}
EOF
echo " VSCode auto-configuré pour uv !"
