from pathlib import Path

p = Path('.github/workflows/sturgeon-operational.yml')
s = p.read_text()
s = s.replace("      - 'forecast_synthesis.py'\n", "      - 'forecast_synthesis.py'\n      - 'forecast_synthesis_engine.py'\n", 1)
s = s.replace("            forecast_synthesis.py \\\n", "            forecast_synthesis.py \\\n            forecast_synthesis_engine.py \\\n", 1)
p.write_text(s)
print('operational workflow now tracks and compiles forecast_synthesis_engine.py')
