from pathlib import Path

p = Path('historical_censored_response_model.py')
s = p.read_text()
s = s.replace('''    if preferred["feature_set"] != "spatial":
        reasons.append("spatial_features_are_not_selected")
''', '')
s = s.replace('''                "at least 75 percent of censored lower bounds satisfied",
                "spatial model selected over amount-only model",
                "manual engineering review and operational hindcast",
''', '''                "at least 75 percent of censored lower bounds satisfied",
                "best-performing feature set selected by out-of-sample validation",
                "manual engineering review and operational hindcast",
''')
p.write_text(s)
print('historical selection rule revised')
