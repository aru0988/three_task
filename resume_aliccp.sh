#!/bin/bash
# Resume AliCCP TC-Prompt experiment from checkpoint
# Run this from the project root: D:\MPT-Rec-three_task\MPT-Rec
#
# Seeds 1-2 complete, Seed 3 has fw+tes done, needs prompt+tcprompt_fixed+tcprompt_learnable

source .venv/Scripts/activate

echo "=== Resuming AliCCP TC-Prompt experiment ==="
echo "Seed 3 (1688738016): running prompt, tcprompt_fixed, tcprompt_learnable"
echo ""

for mode in prompt tcprompt_fixed tcprompt_learnable; do
    echo "=== $(date +%H:%M:%S) seed=1688738016 mode=$mode ==="
    python -u run_aliccp_tcprompt.py --seed 1688738016 --mode $mode --gpu 0 2>&1 | grep -E "RESULT|Error|Traceback|Loading|saved"
done

echo ""
echo "=== All AliCCP experiments complete ==="
echo "Run: python compute_results.py  to calculate mean±std"
