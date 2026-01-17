#!/bin/bash
# Enhanced evaluation monitor with detailed visualization

CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
PURPLE='\033[1;35m'
NC='\033[0m'

clear
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                          🎯 MODEL EVALUATION MONITOR                           ${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Evaluation Tests:${NC}"
echo -e "  • Tactical Skills (5 positions): Win detection, threat blocking"
echo -e "  • Strategic Skills: Center preference, position evaluation"
echo -e "  • Overall Score: 70% tactical + 30% strategic"
echo ""
echo -e "${BLUE}Frequency:${NC} Every 10 iterations (10, 20, 30, ...)"
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

LAST_EVAL=0
EVAL_COUNT=0

while true; do
    # Find latest model
    LATEST_MODEL=$(ls ../../checkpoints/connect4/model_*.pt 2>/dev/null | sed "s/.*model_//" | sed "s/.pt//" | sort -n | tail -1)
    
    if [ ! -z "$LATEST_MODEL" ]; then
        # Check if milestone
        if [ $((LATEST_MODEL % 10)) -eq 0 ] && [ $LATEST_MODEL -gt $LAST_EVAL ]; then
            EVAL_COUNT=$((EVAL_COUNT + 1))
            
            echo ""
            echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
            echo -e "${GREEN}🔍 EVALUATION #$EVAL_COUNT: model_${LATEST_MODEL}.pt${NC}"
            echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
            echo ""
            echo -e "${BLUE}Running comprehensive test battery...${NC}"
            echo ""
            
            # Run evaluation
            /mnt/ssd2pro/miniforge3/envs/tetrisrl/bin/python ../../experiments/evaluate_training.py \
                --model ../../checkpoints/connect4/model_${LATEST_MODEL}.pt
            
            LAST_EVAL=$LATEST_MODEL
            
            echo ""
            echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
            echo -e "${GREEN}✅ Evaluation #$EVAL_COUNT complete for iteration $LATEST_MODEL${NC}"
            echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
            echo ""
        fi
        
        # Status display
        NEXT_EVAL=$(( (LATEST_MODEL / 10 + 1) * 10 ))
        PROGRESS_PCT=$((LATEST_MODEL * 100 / 350))
        EVALS_DONE=$((LATEST_MODEL / 10))
        EVALS_TOTAL=35
        
        # Progress bar (ASCII for compatibility)
        BAR_LENGTH=40
        FILLED=$((PROGRESS_PCT * BAR_LENGTH / 100))
        EMPTY=$((BAR_LENGTH - FILLED))
        # Use simple characters # and -
        BAR=$(printf '%*s' "$FILLED" | tr ' ' '#')$(printf '%*s' "$EMPTY" | tr ' ' '-')
        
        echo -ne "\r\033[K${BLUE}Progress: [$BAR] $LATEST_MODEL/350 ($PROGRESS_PCT%) | Evaluations: $EVALS_DONE/$EVALS_TOTAL | Next: iter $NEXT_EVAL | $(date +%H:%M:%S)${NC}"
    else
        echo -ne "\r\033[K${YELLOW}⏳ Waiting for first checkpoint... | $(date +%H:%M:%S)${NC}"
    fi
    
    sleep 5
done
