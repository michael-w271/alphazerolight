#!/bin/bash
# Comprehensive training monitor with clean single-line updates

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$REPO_ROOT/artifacts/logs/training/training_log_v2.txt"

# Colors
CYAN='\033[1;36m'
PURPLE='\033[1;35m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m'

clear
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                   🎮 CONNECT FOUR MAXIMUM STRENGTH TRAINING                    ${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Progressive Curriculum:${NC}"
echo -e "  • MCTS: 50 → 100 → 200 → 300 → 400 (every 25 iterations)"
echo -e "  • Epochs: 60 → 90 → 120 (at iter 50, 150)"
echo -e "  • Batch: 1024 | Workers: 6"
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

tail -f "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
    # Iteration header
    if [[ $line =~ Iteration\ ([0-9]+)/([0-9]+) ]]; then
        ITER="${BASH_REMATCH[1]}"
        TOTAL="${BASH_REMATCH[2]}"
        echo ""
        echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
        echo -e "${CYAN}📍 ITERATION $ITER/$TOTAL${NC}"
        echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
    
    # Config info
    elif [[ $line =~ MCTS\ searches:\ ([0-9]+) ]]; then
        echo -e "${PURPLE}🔍 MCTS Searches: ${BASH_REMATCH[1]}${NC}"
    elif [[ $line =~ Training\ epochs:\ ([0-9]+) ]]; then
        echo -e "${PURPLE}📚 Training Epochs: ${BASH_REMATCH[1]}${NC}"
    elif [[ $line =~ Temperature:\ ([0-9.]+) ]]; then
        echo -e "${PURPLE}🌡️  Temperature: ${BASH_REMATCH[1]}${NC}"
    elif [[ $line =~ Phase:\ ([a-z]+) ]]; then
        echo -e "${BLUE}🎮 Phase: ${BASH_REMATCH[1]}${NC}"
    
    # Self-play progress - UPDATE IN PLACE (single line)
    elif [[ $line =~ Self-play:\ +([0-9]+)%\|.*\|\ +([0-9]+)/([0-9]+)\ \[([0-9:]+)\<([0-9:]+),\ +([0-9.]+)s/it\] ]]; then
        PCT="${BASH_REMATCH[1]}"
        DONE="${BASH_REMATCH[2]}"
        TOTAL="${BASH_REMATCH[3]}"
        ELAPSED="${BASH_REMATCH[4]}"
        REMAIN="${BASH_REMATCH[5]}"
        SPEED="${BASH_REMATCH[6]}"
        echo -ne "\r\033[K${BLUE}🎲 Self-play: $DONE/$TOTAL games ($PCT%) | ${SPEED}s/game | ETA: ${REMAIN}${NC}"
    
    # Training progress - UPDATE IN PLACE (single line)
    elif [[ $line =~ Training:\ +([0-9]+)%\|.*\|\ +([0-9]+)/([0-9]+)\ \[([0-9:]+)\<([0-9:]+),\ +([0-9.]+)it/s\] ]]; then
        PCT="${BASH_REMATCH[1]}"
        DONE="${BASH_REMATCH[2]}"
        TOTAL="${BASH_REMATCH[3]}"
        ELAPSED="${BASH_REMATCH[4]}"
        REMAIN="${BASH_REMATCH[5]}"
        SPEED="${BASH_REMATCH[6]}"
        echo -ne "\r\033[K${BLUE}🧠 Training NN: $DONE/$TOTAL epochs ($PCT%) | ${SPEED} it/s | ETA: ${REMAIN}${NC}"
    
    # Sampled game types
    elif [[ $line =~ Sampled\ game\ types ]]; then
        echo ""  # New line after progress
        echo -e "${PURPLE}$line${NC}"
    
    # Generated samples
    elif [[ $line =~ Generated\ ([0-9]+)\ training\ samples ]]; then
        SAMPLES="${BASH_REMATCH[1]}"
        echo -e "${GREEN}✅ Generated $SAMPLES training samples${NC}"
    
    # Game outcomes
    elif [[ $line =~ Game\ outcomes:.*AI\ wins\ ([0-9]+).*Opponent\ wins\ ([0-9]+).*Draws\ ([0-9]+) ]]; then
        WINS="${BASH_REMATCH[1]}"
        LOSSES="${BASH_REMATCH[2]}"
        DRAWS="${BASH_REMATCH[3]}"
        echo -e "${GREEN}   Win rate: $WINS wins, $LOSSES losses, $DRAWS draws${NC}"
    
    # Loss metrics
    elif [[ $line =~ Loss:\ ([0-9.]+).*Policy:\ ([0-9.]+).*Value:\ ([0-9.]+) ]]; then
        TOTAL_LOSS="${BASH_REMATCH[1]}"
        POLICY="${BASH_REMATCH[2]}"
        VALUE="${BASH_REMATCH[3]}"
        echo ""  # New line after training progress
        echo -e "${YELLOW}📊 Loss: ${TOTAL_LOSS} (Policy: ${POLICY}, Value: ${VALUE})${NC}"
    
    # Saving checkpoint
    elif [[ $line =~ Saving\ checkpoint ]]; then
        echo -e "${GREEN}💾 Saving checkpoint...${NC}"
    
    # Iteration complete
    elif [[ $line =~ complete.*took\ ([0-9.]+)min ]]; then
        TIME="${BASH_REMATCH[1]}"
        echo -e "${GREEN}✅ Iteration complete! (${TIME} min)${NC}"
    
    # Overall progress with ETA
    elif [[ $line =~ Overall\ Progress:.*([0-9]+)/([0-9]+).*ETA=([0-9a-z:]+) ]]; then
        DONE="${BASH_REMATCH[1]}"
        TOTAL="${BASH_REMATCH[2]}"
        ETA="${BASH_REMATCH[3]}"
        PCT=$((DONE * 100 / TOTAL))
        echo -e "${BLUE}⏱️  Overall: $DONE/$TOTAL iterations ($PCT%) | ETA: $ETA remaining${NC}"
        echo ""
    
    # Skip these
    elif [[ $line =~ ^[[:space:]]*$ ]] || \
         [[ $line =~ ^=+$ ]] || \
         [[ $line =~ CUDA.*compatibility ]] || \
         [[ $line =~ size\ mismatch ]]; then
        continue
    
    # Bootstrap/Main phase announcements
    elif [[ $line =~ BOOTSTRAP ]] || [[ $line =~ MAIN ]]; then
        echo -e "${BLUE}$line${NC}"
    fi
done
