import streamlit as st
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path - go up from ui/ to src/
current_dir = Path(__file__).parent  # ui/
src_dir = current_dir.parent.parent  # src/
sys.path.insert(0, str(src_dir))

from alpha_zero_light.game.tictactoe import TicTacToe
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS

# Page config
st.set_page_config(
    page_title="AlphaZero Tic-Tac-Toe",
    page_icon="🎮",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 80px;
        font-size: 32px;
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid #ddd;
    }
    .player-x {
        color: #2196F3;
    }
    .player-o {
        color: #f44336;
    }
    h1 {
        text-align: center;
        background: linear-gradient(90deg, #2196F3, #f44336);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .game-status {
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        padding: 1em;
        border-radius: 10px;
        margin: 1em 0;
    }
    .status-ongoing {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .status-win {
        background-color: #c8e6c9;
        color: #388e3c;
    }
    .status-draw {
        background-color: #fff9c4;
        color: #f57c00;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and initialize MCTS"""
    game = TicTacToe()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNet(game, num_res_blocks=4, num_hidden=64).to(device)
    
    # Load the latest checkpoint
    checkpoint_dir = Path(__file__).parent.parent.parent.parent.parent / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("model_*.pt"))
    
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
        model.eval()
        st.sidebar.success(f"✅ Loaded: {latest_checkpoint.name}")
    else:
        st.sidebar.warning("⚠️ No checkpoint found, using untrained model")
    
    args = {
        'C': 2,
        'num_searches': 100,  # More searches for better play
    }
    
    mcts = MCTS(game, args, model)
    
    return game, model, mcts, device

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'board' not in st.session_state:
        st.session_state.board = np.zeros((3, 3))
    if 'current_player' not in st.session_state:
        st.session_state.current_player = 1  # Human is 1 (X)
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False
    if 'winner' not in st.session_state:
        st.session_state.winner = None
    if 'last_ai_probs' not in st.session_state:
        st.session_state.last_ai_probs = None

def reset_game():
    """Reset the game state"""
    st.session_state.board = np.zeros((3, 3))
    st.session_state.current_player = 1
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.last_ai_probs = None

def get_cell_display(value):
    """Get display character for cell value"""
    if value == 1:
        return "X"
    elif value == -1:
        return "O"
    return ""

def check_game_over(game, board, last_action):
    """Check if game is over and return status"""
    if last_action is not None:
        value, is_terminal = game.get_value_and_terminated(board, last_action)
        if is_terminal:
            if value == 1:
                return True, st.session_state.current_player
            else:
                return True, 0  # Draw
    return False, None

def ai_move(game, mcts):
    """Make AI move"""
    if st.session_state.game_over:
        return
    
    # Get AI's perspective of the board
    ai_state = game.change_perspective(st.session_state.board.copy(), player=-1)
    
    # Get action probabilities from MCTS
    action_probs = mcts.search(ai_state)
    
    # Store for visualization
    st.session_state.last_ai_probs = action_probs.reshape(3, 3)
    
    # Choose best action
    valid_moves = game.get_valid_moves(st.session_state.board)
    action_probs *= valid_moves
    action = np.argmax(action_probs)
    
    # Make move
    st.session_state.board = game.get_next_state(
        st.session_state.board, 
        action, 
        st.session_state.current_player
    )
    
    # Check if game is over
    is_over, winner = check_game_over(game, st.session_state.board, action)
    if is_over:
        st.session_state.game_over = True
        st.session_state.winner = winner
    else:
        # Switch player
        st.session_state.current_player = game.get_opponent(st.session_state.current_player)

def make_move(row, col, game, mcts):
    """Handle human move"""
    if st.session_state.game_over:
        return
    
    if st.session_state.board[row, col] != 0:
        st.warning("⚠️ Cell already occupied!")
        return
    
    # Make human move
    action = row * 3 + col
    st.session_state.board = game.get_next_state(
        st.session_state.board,
        action,
        st.session_state.current_player
    )
    
    # Check if game is over after human move
    is_over, winner = check_game_over(game, st.session_state.board, action)
    if is_over:
        st.session_state.game_over = True
        st.session_state.winner = winner
        return
    
    # Switch to AI
    st.session_state.current_player = game.get_opponent(st.session_state.current_player)
    
    # Make AI move
    ai_move(game, mcts)

def main():
    st.title("🎮 AlphaZero Tic-Tac-Toe")
    
    # Load model
    game, model, mcts, device = load_model()
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    st.sidebar.title("ℹ️ Game Info")
    st.sidebar.markdown("""
    **How to Play:**
    - You are **X** (Blue)
    - AI is **O** (Red)
    - Click on empty cells to make your move
    - AI will respond automatically
    """)
    
    st.sidebar.markdown(f"**Device:** {device}")
    
    # Game status
    if st.session_state.game_over:
        if st.session_state.winner == 1:
            status_class = "status-win"
            status_text = "🎉 You Win! (X)"
        elif st.session_state.winner == -1:
            status_class = "status-win"
            status_text = "🤖 AI Wins! (O)"
        else:
            status_class = "status-draw"
            status_text = "🤝 It's a Draw!"
    else:
        status_class = "status-ongoing"
        current_symbol = "X" if st.session_state.current_player == 1 else "O"
        current_name = "Your" if st.session_state.current_player == 1 else "AI's"
        status_text = f"🎯 {current_name} Turn ({current_symbol})"
    
    st.markdown(f'<div class="game-status {status_class}">{status_text}</div>', unsafe_allow_html=True)
    
    # Game board
    st.markdown("### Game Board")
    for row in range(3):
        cols = st.columns(3)
        for col in range(3):
            cell_value = st.session_state.board[row, col]
            cell_display = get_cell_display(cell_value)
            
            # Color code the button based on player
            if cell_value == 1:
                button_label = f":blue[{cell_display}]"
            elif cell_value == -1:
                button_label = f":red[{cell_display}]"
            else:
                button_label = " "
            
            with cols[col]:
                if st.button(
                    button_label,
                    key=f"cell_{row}_{col}",
                    disabled=st.session_state.game_over or cell_value != 0 or st.session_state.current_player == -1,
                    use_container_width=True
                ):
                    make_move(row, col, game, mcts)
                    st.rerun()
    
    # New game button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 New Game", use_container_width=True, type="primary"):
            reset_game()
            st.rerun()
    
    # MCTS Visualization
    if st.session_state.last_ai_probs is not None:
        st.markdown("---")
        st.markdown("### 🧠 AI's Last Move Analysis")
        st.markdown("**MCTS Visit Distribution** (higher = more explored)")
        
        # Create a heatmap-like display
        cols = st.columns(3)
        max_prob = np.max(st.session_state.last_ai_probs)
        
        for row in range(3):
            for col in range(3):
                prob = st.session_state.last_ai_probs[row, col]
                percentage = prob * 100
                
                # Color intensity based on probability
                if prob > 0:
                    intensity = int((prob / max_prob) * 255) if max_prob > 0 else 0
                    color = f"rgba(33, 150, 243, {prob/max_prob:.2f})"
                else:
                    color = "rgba(200, 200, 200, 0.3)"
                
                with cols[col]:
                    st.markdown(
                        f'<div style="background-color: {color}; padding: 15px; '
                        f'border-radius: 8px; text-align: center; font-weight: bold; '
                        f'border: 2px solid #ddd; margin: 2px;">'
                        f'{percentage:.1f}%</div>',
                        unsafe_allow_html=True
                    )
            cols = st.columns(3)  # Reset for next row

if __name__ == "__main__":
    main()
