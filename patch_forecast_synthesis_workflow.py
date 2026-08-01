#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


path = Path('.github/workflows/sturgeon-operational.yml')
text = path.read_text()

# Trigger paths.
anchor = "      - 'forecast_impacts_v2.py'\n"
addition = (
    "      - 'forecast_impacts_v2.py'\n"
    "      - 'forecast_synthesis.py'\n"
    "      - 'historical_response_shadow_forecast.py'\n"
)
if "      - 'forecast_synthesis.py'\n" not in text:
    text = replace_once(text, anchor, addition, "trigger paths")

# Compile check.
anchor = "            forecast_impacts_v2.py\n"
addition = (
    "            forecast_impacts_v2.py \\\n"
    "            historical_response_shadow_forecast.py \\\n"
    "            forecast_synthesis.py\n"
)
if "            forecast_synthesis.py\n" not in text:
    text = replace_once(text, anchor, addition, "compile scripts")

# Same-cycle shadow and synthesis, immediately after authoritative readiness.
anchor = '''          echo "$CODE" > sturgeon_pipeline_output/logs/operational_readiness.exitcode
          exit "$CODE"
      - name: Screen new calibration candidates
'''
addition = '''          echo "$CODE" > sturgeon_pipeline_output/logs/operational_readiness.exitcode
          exit "$CODE"
      - name: Build current-cycle historical response shadow
        continue-on-error: true
        run: |
          mkdir -p sturgeon_pipeline_output/logs sturgeon_pipeline_output/diagnostics
          set +e
          set -o pipefail
          python historical_response_shadow_forecast.py \\
            --forecast sturgeon_pipeline_output/forecast_v2/forecast_impacts_v2.json \\
            --model output/historical_event_backfill/historical_censored_response_model.json \\
            --output sturgeon_pipeline_output/diagnostics/historical_response_shadow_current.json \\
            2>&1 | tee sturgeon_pipeline_output/logs/historical_response_shadow_current.log
          CODE=${PIPESTATUS[0]}
          echo "$CODE" > sturgeon_pipeline_output/logs/historical_response_shadow_current.exitcode
          exit "$CODE"
      - name: Synthesize coherent operational forecast
        continue-on-error: true
        env:
          TRIGGER_REASON: ${{ inputs.trigger_reason || 'code change' }}
        run: |
          mkdir -p sturgeon_pipeline_output/logs
          if [ "$(cat sturgeon_pipeline_output/logs/operational_readiness.exitcode 2>/dev/null || echo 1)" != "0" ]; then
            echo 'Skipping forecast synthesis because construction readiness failed.' | tee sturgeon_pipeline_output/logs/forecast_synthesis.log
            echo 3 > sturgeon_pipeline_output/logs/forecast_synthesis.exitcode
            exit 3
          fi
          set +e
          set -o pipefail
          python forecast_synthesis.py 2>&1 | tee sturgeon_pipeline_output/logs/forecast_synthesis.log
          CODE=${PIPESTATUS[0]}
          echo "$CODE" > sturgeon_pipeline_output/logs/forecast_synthesis.exitcode
          exit "$CODE"
      - name: Screen new calibration candidates
'''
if "      - name: Synthesize coherent operational forecast\n" not in text:
    text = replace_once(text, anchor, addition, "synthesis steps")

# Require synthesis in the operational exit-code loop.
anchor = '''            project_threshold_ensemble \\
            operational_readiness; do
'''
addition = '''            project_threshold_ensemble \\
            operational_readiness \\
            forecast_synthesis; do
'''
if "            forecast_synthesis; do\n" not in text:
    text = replace_once(text, anchor, addition, "required synthesis exit code")

# Report current-cycle shadow as a diagnostic.
anchor = "          for name in assimilation_candidates storage_state_candidate; do\n"
addition = "          for name in assimilation_candidates storage_state_candidate historical_response_shadow_current; do\n"
if "historical_response_shadow_current; do" not in text:
    text = replace_once(text, anchor, addition, "shadow diagnostic loop")

# Add synthesis and manifest to integrity checks.
anchor = "              'uncertainty': root / 'diagnostics/uncertainty_sensitivity.json',\n"
addition = (
    "              'uncertainty': root / 'diagnostics/uncertainty_sensitivity.json',\n"
    "              'synthesis': root / 'forecast_v2/forecast_synthesis.json',\n"
    "              'manifest': root / 'run_manifest.json',\n"
)
if "              'synthesis': root / 'forecast_v2/forecast_synthesis.json',\n" not in text:
    text = replace_once(text, anchor, addition, "integrity files")

anchor = "          if data['uncertainty'].get('status') != 'operational_sensitivity_not_calibrated_probability':\n              raise RuntimeError('uncertainty sensitivity output is unavailable')\n"
addition = """          if data['uncertainty'].get('status') != 'operational_sensitivity_not_calibrated_probability':
              raise RuntimeError('uncertainty sensitivity output is unavailable')
          if data['synthesis'].get('status') != 'operational_forecast_synthesis':
              raise RuntimeError('forecast synthesis is unavailable')
          if data['synthesis'].get('run_id') != data['manifest'].get('run_id'):
              raise RuntimeError('synthesis and manifest run IDs do not match')
          if data['manifest'].get('cycle_consistency', {}).get('status') not in {
              'consistent', 'operational_consistent_shadow_pending'
          }:
              raise RuntimeError('run manifest cycle consistency failed')
"""
if "forecast synthesis is unavailable" not in text:
    text = replace_once(text, anchor, addition, "synthesis integrity")

# Atomic attempt/valid/latest publication. A failed run must not replace the authoritative package.
old = '''      - name: Prepare latest operational output
        if: ${{ always() && !cancelled() }}
        run: |
          rm -rf output/latest
          mkdir -p output/latest
          if [ -d sturgeon_pipeline_output ]; then
            rsync -a --exclude='sturgeon_pipeline_latest.zip' sturgeon_pipeline_output/ output/latest/
          fi
          date -u +'%Y-%m-%dT%H:%M:%SZ' > output/latest/workflow_completed_utc.txt
          if [ "$(cat output/latest/logs/operational_validation.exitcode 2>/dev/null || echo 1)" = "0" ]; then
            echo operational > output/latest/workflow_mode.txt
          else
            echo degraded > output/latest/workflow_mode.txt
          fi
      - name: Upload operational package
        if: ${{ always() && !cancelled() }}
        uses: actions/upload-artifact@v4
        with:
          name: sturgeon-operational-latest
          path: output/latest
          retention-days: 14
          if-no-files-found: warn
      - name: Commit latest operational outputs
        if: ${{ always() && !cancelled() }}
        run: |
          git config user.name 'github-actions[bot]'
          git config user.email '41898282+github-actions[bot]@users.noreply.github.com'

          rm -rf /tmp/sturgeon-operational-latest
          cp -a output/latest /tmp/sturgeon-operational-latest

          for attempt in 1 2 3 4; do
            echo "Publish attempt ${attempt}"
            git fetch origin main
            git reset --hard origin/main
            rm -rf output/latest
            cp -a /tmp/sturgeon-operational-latest output/latest
            git add -f output/latest

            if git diff --cached --quiet; then
              echo 'No operational output changes to publish.'
              break
            fi

            git commit -m 'Update Sturgeon operational forecast [skip ci]'
            if git push origin HEAD:main; then
              break
            fi

            if [ "$attempt" = "4" ]; then
              echo 'Unable to publish operational output after four attempts.' >&2
              exit 1
            fi
            echo 'Another run updated main first; retrying from the new branch head.'
            sleep $((attempt * 2))
          done
      - name: Enforce operational validation
        if: ${{ always() && !cancelled() }}
        run: |
          test "$(cat output/latest/logs/operational_validation.exitcode 2>/dev/null || echo 1)" = "0"
'''
new = '''      - name: Prepare attempted and validated operational outputs
        if: ${{ always() && !cancelled() }}
        run: |
          rm -rf output/latest_attempt
          mkdir -p output/latest_attempt
          if [ -d sturgeon_pipeline_output ]; then
            rsync -a --exclude='sturgeon_pipeline_latest.zip' sturgeon_pipeline_output/ output/latest_attempt/
          fi
          date -u +'%Y-%m-%dT%H:%M:%SZ' > output/latest_attempt/workflow_completed_utc.txt
          if [ "$(cat output/latest_attempt/logs/operational_validation.exitcode 2>/dev/null || echo 1)" = "0" ]; then
            echo operational > output/latest_attempt/workflow_mode.txt
            rm -rf output/latest output/latest_valid
            cp -a output/latest_attempt output/latest
            cp -a output/latest_attempt output/latest_valid
            mkdir -p output/history
            cp output/latest_attempt/history/forecast_history.csv output/history/forecast_history.csv
          else
            echo degraded > output/latest_attempt/workflow_mode.txt
            echo 'Attempt failed validation; authoritative latest and latest_valid are unchanged.'
          fi
      - name: Upload operational attempt
        if: ${{ always() && !cancelled() }}
        uses: actions/upload-artifact@v4
        with:
          name: sturgeon-operational-attempt
          path: output/latest_attempt
          retention-days: 14
          if-no-files-found: warn
      - name: Commit attempted and validated operational outputs
        if: ${{ always() && !cancelled() }}
        run: |
          git config user.name 'github-actions[bot]'
          git config user.email '41898282+github-actions[bot]@users.noreply.github.com'

          rm -rf /tmp/sturgeon-operational-attempt /tmp/sturgeon-operational-valid /tmp/sturgeon-operational-latest /tmp/sturgeon-forecast-history.csv
          cp -a output/latest_attempt /tmp/sturgeon-operational-attempt
          VALID=0
          if [ "$(cat output/latest_attempt/logs/operational_validation.exitcode 2>/dev/null || echo 1)" = "0" ]; then
            VALID=1
            cp -a output/latest_valid /tmp/sturgeon-operational-valid
            cp -a output/latest /tmp/sturgeon-operational-latest
            cp output/history/forecast_history.csv /tmp/sturgeon-forecast-history.csv
          fi

          for attempt in 1 2 3 4; do
            echo "Publish attempt ${attempt}"
            git fetch origin main
            git reset --hard origin/main
            rm -rf output/latest_attempt
            cp -a /tmp/sturgeon-operational-attempt output/latest_attempt
            git add -f output/latest_attempt
            if [ "$VALID" = "1" ]; then
              rm -rf output/latest output/latest_valid
              cp -a /tmp/sturgeon-operational-latest output/latest
              cp -a /tmp/sturgeon-operational-valid output/latest_valid
              mkdir -p output/history
              cp /tmp/sturgeon-forecast-history.csv output/history/forecast_history.csv
              git add -f output/latest output/latest_valid output/history/forecast_history.csv
            fi

            if git diff --cached --quiet; then
              echo 'No operational output changes to publish.'
              break
            fi

            git commit -m 'Update Sturgeon operational forecast [skip ci]'
            if git push origin HEAD:main; then
              break
            fi

            if [ "$attempt" = "4" ]; then
              echo 'Unable to publish operational output after four attempts.' >&2
              exit 1
            fi
            echo 'Another run updated main first; retrying from the new branch head.'
            sleep $((attempt * 2))
          done
      - name: Enforce operational validation
        if: ${{ always() && !cancelled() }}
        run: |
          test "$(cat output/latest_attempt/logs/operational_validation.exitcode 2>/dev/null || echo 1)" = "0"
'''
if "Prepare attempted and validated operational outputs" not in text:
    text = replace_once(text, old, new, "atomic publication")

path.write_text(text)
print('Operational workflow now synthesizes one coherent forecast and publishes atomically.')
