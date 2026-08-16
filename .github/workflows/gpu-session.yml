name: Kaggle GPU Session

on:
  workflow_dispatch:
    inputs:
      session_id:
        description: "GPU session ID"
        required: true
        type: string

      api_url:
        description: "Railway API URL"
        required: true
        type: string

      worker_token:
        description: "Worker token"
        required: true
        type: string

jobs:
  gpu-session:
    runs-on: ubuntu-latest
    timeout-minutes: 25

    env:
      SESSION_ID: ${{ inputs.session_id }}
      API_URL: ${{ inputs.api_url }}
      WORKER_TOKEN: ${{ inputs.worker_token }}

      KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
      KAGGLE_API_TOKEN: ${{ secrets.KAGGLE_API_TOKEN }}

    steps:

      # ------------------------------------------------
      # CHECKOUT
      # ------------------------------------------------

      - name: Checkout
        uses: actions/checkout@v4

      # ------------------------------------------------
      # PYTHON
      # ------------------------------------------------

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      # ------------------------------------------------
      # INSTALL
      # ------------------------------------------------

      - name: Install dependencies
        shell: bash
        run: |
          set -e

          python -m pip install --upgrade pip

          python -m pip install \
            kaggle \
            requests

          kaggle --version

      # ------------------------------------------------
      # VALIDATION
      # ------------------------------------------------

      - name: Validate configuration
        shell: bash
        run: |
          set -e

          echo "========================================"
          echo "VALIDATION"
          echo "========================================"

          echo "Session: ${SESSION_ID}"
          echo "API: ${API_URL}"
          echo "Token length: ${#WORKER_TOKEN}"

          test -n "${SESSION_ID}"
          test -n "${API_URL}"
          test -n "${WORKER_TOKEN}"

          test -n "${KAGGLE_USERNAME}"
          test -n "${KAGGLE_API_TOKEN}"

          echo "Validation OK."

      # ------------------------------------------------
      # PREPARE KAGGLE SCRIPT
      # ------------------------------------------------

      - name: Prepare Kaggle worker
        shell: bash
        run: |
          set -e

          echo "========================================"
          echo "PREPARING KAGGLE WORKER"
          echo "========================================"

          mkdir -p kaggle_upload

          cp kaggle_worker/script.py kaggle_upload/script.py

          python - <<'PY'
          from pathlib import Path

          p = Path("kaggle_upload/script.py")

          text = p.read_text()

          replacements = {
              "%%SESSION_ID%%": "${{ inputs.session_id }}",
              "%%API_URL%%": "${{ inputs.api_url }}",
              "%%WORKER_TOKEN%%": "${{ inputs.worker_token }}",
          }

          for old, new in replacements.items():
              text = text.replace(old, new)

          p.write_text(text)

          if "%%SESSION_ID%%" in text:
              raise SystemExit("SESSION_ID placeholder remains")

          if "%%API_URL%%" in text:
              raise SystemExit("API_URL placeholder remains")

          if "%%WORKER_TOKEN%%" in text:
              raise SystemExit("WORKER_TOKEN placeholder remains")

          print("All placeholders replaced.")
          PY

          python -m py_compile kaggle_upload/script.py

          echo "Worker Python syntax OK."

      # ------------------------------------------------
      # KAGGLE METADATA
      # ------------------------------------------------

      - name: Create Kaggle metadata
        shell: bash
        run: |
          set -e

          mkdir -p kaggle_upload

          cat > kaggle_upload/kernel-metadata.json <<EOF
          {
            "id": "${KAGGLE_USERNAME}/gpu-session-${SESSION_ID}",
            "title": "GPU Session ${SESSION_ID}",
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": true,
            "enable_gpu": true,
            "enable_tpu": false,
            "enable_internet": true
          }
          EOF

          cat kaggle_upload/kernel-metadata.json

      # ------------------------------------------------
      # PUSH KAGGLE
      # ------------------------------------------------

      - name: Push Kaggle GPU worker
        shell: bash
        working-directory: kaggle_upload
        run: |
          set -e

          echo "========================================"
          echo "PUSHING KAGGLE"
          echo "========================================"

          kaggle kernels push

          echo "Kaggle worker submitted."

      # ------------------------------------------------
      # WAIT FOR WORKER
      # ------------------------------------------------

      - name: Wait for GPU worker
        shell: bash
        run: |
          set +e

          echo "======================================"
          echo "WAITING FOR GPU WORKER"
          echo "======================================"

          READY=false

          for i in $(seq 1 180); do

            echo ""
            echo "CHECK $i / 180"

            RESPONSE=$(curl -sS \
              --max-time 20 \
              "${API_URL}/gpu/session/${SESSION_ID}" \
              || true)

            echo "$RESPONSE"

            STATUS=$(echo "$RESPONSE" | python -c '
          import sys,json
          try:
              print(json.load(sys.stdin).get("status",""))
          except:
              print("")
          ')

            echo "Current status: $STATUS"

            if [ "$STATUS" = "active" ]; then
              READY=true

              echo ""
              echo "======================================"
              echo "GPU WORKER IS READY"
              echo "======================================"

              break
            fi

            if [ "$STATUS" = "failed" ] ||
               [ "$STATUS" = "error" ] ||
               [ "$STATUS" = "expired" ] ||
               [ "$STATUS" = "stopped" ]; then

              echo "GPU worker failed: $STATUS"
              exit 1
            fi

            sleep 5

          done

          if [ "$READY" != "true" ]; then
            echo "GPU worker did not become ready."
            exit 1
          fi

      # ------------------------------------------------
      # GITHUB KEEP ALIVE
      # ------------------------------------------------

      - name: Keep GitHub session alive
        shell: bash
        run: |
          set +e

          echo "======================================"
          echo "GITHUB KEEP-ALIVE LOOP"
          echo "======================================"

          for i in $(seq 1 240); do

            echo ""
            echo "KEEP-ALIVE CHECK $i / 240"

            RESPONSE=$(curl -sS \
              --max-time 20 \
              "${API_URL}/gpu/session/${SESSION_ID}" \
              || true)

            echo "$RESPONSE"

            STATUS=$(echo "$RESPONSE" | python -c '
          import sys,json
          try:
              print(json.load(sys.stdin).get("status",""))
          except:
              print("")
          ')

            echo "STATUS: $STATUS"

            case "$STATUS" in

              active)
                echo "GPU worker is active."
                ;;

              starting)
                echo "Worker still starting."
                ;;

              stopped|expired|completed|failed|error)
                echo "Session ended: $STATUS"
                exit 0
                ;;

              *)
                echo "Unknown status. Continuing."
                ;;

            esac

            sleep 5

          done

          echo "Keep-alive period finished."

      # ------------------------------------------------
      # ARTIFACT
      # ------------------------------------------------

      - name: Save worker files
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: gpu-worker-${{ inputs.session_id }}
          path: |
            kaggle_upload/script.py
            kaggle_upload/kernel-metadata.json
          if-no-files-found: ignore
