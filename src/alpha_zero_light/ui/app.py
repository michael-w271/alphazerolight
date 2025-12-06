import streamlit as st
import torch
import numpy as np
import sys
import os
import json
from pathlib import Path

# Add src to path - go up from ui/ to src/
current_dir = Path(__file__).parent  # ui/
src_dir = current_dir.parent.parent  # src/
sys.path.insert(0, str(src_dir))

from alpha_zero_light.game.tictactoe import TicTacToe
from alpha_zero_light.game.gomoku import Gomoku
from alpha_zero_light.game.gomoku_9x9 import Gomoku9x9
from alpha_zero_light.model.network import ResNet
from alpha_zero_light.mcts.mcts import MCTS

# Page config
st.set_page_config(
    page_title="AlphaZero Game Arena",
    page_icon="🎮",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* General Button Styling */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        border: 1px solid #ccc;
        transition: all 0.2s;
    }
    
    /* TicTacToe Specific */
    .tictactoe-btn {
        height: 80px !important;
        font-size: 32px !important;
        font-weight: bold;
    }
    
    /* Gomoku Specific */
    .gomoku-btn {
        height: 30px !important;
        width: 30px !important;
        padding: 0px !important;
        font-size: 20px !important;
        line-height: 30px !important;
        border-radius: 50% !important; /* Circular buttons */
        border: 1px solid #888 !important;
        background-color: #e6b800 !important; /* Wood-like color */
    }
    
    /* Player Colors */
    .player-x { color: #2196F3; }
    .player-o { color: #f44336; }
    
    /* Gomoku Stones */
    .gomoku-black {
        color: black !important;
        background-color: black !important;
        border: 1px solid white !important;
        box-shadow: 2px 2px 2px rgba(0,0,0,0.3);
    }
    .gomoku-white {
        color: white !important;
        background-color: white !important;
        border: 1px solid #ccc !important;
        box-shadow: 2px 2px 2px rgba(0,0,0,0.3);
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
    .status-ongoing { background-color: #e3f2fd; color: #1976d2; }
    .status-win { background-color: #c8e6c9; color: #388e3c; }
    .status-draw { background-color: #fff9c4; color: #f57c00; }
    
    /* Compact columns for Gomoku */
    div[data-testid="stHorizontalBlock"] > div {
        min-width: 0px !important;
        padding: 1px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(game_name, checkpoint_name=None):
    """Load the trained model and initialize MCTS"""
    if game_name == "TicTacToe":
        game = TicTacToe()
        num_res_blocks = 4
        num_hidden = 64
    elif game_name == "Gomoku 9x9":
        game = Gomoku9x9()
        num_res_blocks = 4
        num_hidden = 64
    elif game_name == "Gomoku 9x9 GPU Test":
        game = Gomoku9x9()
        num_res_blocks = 15  # GPU test model
        num_hidden = 384
    elif game_name == "Gomoku Long":
        # Currently training model - loads latest checkpoint
        game = Gomoku9x9()
        num_res_blocks = 9
        num_hidden = 256
        checkpoint_dir = Path(__file__).parent.parent.parent.parent.parent / "checkpoints" / "gomoku_long"
    elif game_name == "Gomoku Fixed":
        # NEW: Fixed model with UCB + edge win fixes
        game = Gomoku9x9()
        num_res_blocks = 12
        num_hidden = 384
        checkpoint_dir = Path(__file__).parent.parent.parent.parent.parent / "checkpoints" / "gomoku_fixed"
    else:  # Gomoku 15x15
        game = Gomoku()
        num_res_blocks = 8 
        num_hidden = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNet(game, num_res_blocks=num_res_blocks, num_hidden=num_hidden).to(device)
    
    # Determine checkpoint directory
    if 'checkpoint_dir' not in locals():
        checkpoint_dir = Path(__file__).parent.parent.parent.parent.parent / "checkpoints" / game_name.lower().replace(" ", "_")
        if not checkpoint_dir.exists():
            checkpoint_dir = Path(__file__).parent.parent.parent.parent.parent / "checkpoints"
    
    # Get all available checkpoints
    checkpoints = sorted(checkpoint_dir.glob("model_*.pt"))
    
    if checkpoints:
        # Use provided checkpoint name or default to latest
        if checkpoint_name and checkpoint_name != "Latest (Auto)":
            latest_checkpoint = checkpoint_dir / checkpoint_name
        else:
            latest_checkpoint = checkpoints[-1]
        
        try:
            model.load_state_dict(torch.load(latest_checkpoint, map_location=device), strict=False)
            model.eval()
            st.sidebar.success(f"✅ Loaded: {latest_checkpoint.name}")
        except Exception as e:
             st.sidebar.warning(f"⚠️ Could not load checkpoint: {e}")
             st.sidebar.info("Using untrained model")
    else:
        st.sidebar.warning("⚠️ No checkpoint found, using untrained model")
    
    args = {
        'C': 2,
        'num_searches': 100 if game_name == "TicTacToe" else (200 if "9x9" in game_name else 400),
    }
    
    mcts = MCTS(game, args, model)
    
    return game, model, mcts, device

def initialize_session_state(game):
    """Initialize Streamlit session state"""
    if 'board' not in st.session_state or st.session_state.board.shape != (game.row_count, game.column_count):
        st.session_state.board = game.get_initial_state()
    if 'current_player' not in st.session_state:
        st.session_state.current_player = 1
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False
    if 'winner' not in st.session_state:
        st.session_state.winner = None
    if 'last_ai_probs' not in st.session_state:
        st.session_state.last_ai_probs = None

def reset_game(game):
    """Reset the game state"""
    st.session_state.board = game.get_initial_state()
    st.session_state.current_player = 1
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.last_ai_probs = None

def get_cell_display(value, game_name):
    """Get display character for cell value"""
    if game_name == "TicTacToe":
        if value == 1: return "X"
        elif value == -1: return "O"
    else:
        return ""
    return ""

def check_game_over(game, board, last_action):
    """Check if game is over and return status"""
    if last_action is not None:
        value, is_terminal = game.get_value_and_terminated(board, last_action)
        if is_terminal:
            if value == 1:
                return True, st.session_state.current_player
            else:
                return True, 0
    return False, None

def ai_move(game, mcts):
    """Make AI move"""
    if st.session_state.game_over:
        return
    
    ai_state = game.change_perspective(st.session_state.board.copy(), player=-1)
    
    with st.spinner("AI is thinking..."):
        action_probs = mcts.search(ai_state)
    
    st.session_state.last_ai_probs = action_probs.reshape(game.row_count, game.column_count)
    
    valid_moves = game.get_valid_moves(st.session_state.board)
    action_probs *= valid_moves
    action = np.argmax(action_probs)
    
    st.session_state.board = game.get_next_state(
        st.session_state.board,
        action,
        st.session_state.current_player
    )
    
    is_over, winner = check_game_over(game, st.session_state.board, action)
    if is_over:
        st.session_state.game_over = True
        st.session_state.winner = winner
    else:
        st.session_state.current_player = game.get_opponent(st.session_state.current_player)

def make_move(row, col, game, mcts):
    """Handle human move"""
    if st.session_state.game_over:
        return
    
    if st.session_state.board[row, col] != 0:
        st.warning("⚠️ Cell already occupied!")
        return
    
    action = row * game.column_count + col
    st.session_state.board = game.get_next_state(
        st.session_state.board,
        action,
        st.session_state.current_player
    )
    
    is_over, winner = check_game_over(game, st.session_state.board, action)
    if is_over:
        st.session_state.game_over = True
        st.session_state.winner = winner
        return
    
    st.session_state.current_player = game.get_opponent(st.session_state.current_player)
    ai_move(game, mcts)
    """Initialize Streamlit session state"""
    if 'board' not in st.session_state or st.session_state.board.shape != (game.row_count, game.column_count):
        board = game.get_initial_state()
        # Convert to numpy if it's a tensor
        if hasattr(board, 'cpu'):
            board = board.cpu().numpy()
        if board.ndim == 3:  # (1, H, W) -> (H, W)
            board = board.squeeze(0)
        st.session_state.board = board
    if 'current_player' not in st.session_state:
        st.session_state.current_player = 1  # Human is 1 (Black/X)
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False
    if 'winner' not in st.session_state:
        st.session_state.winner = None
    if 'last_ai_probs' not in st.session_state:
        st.session_state.last_ai_probs = None

def get_board_for_display(board):
    """Convert board to numpy array for display"""
    if hasattr(board, 'cpu'):
        board = board.cpu().numpy()
    if board.ndim == 3:
        board = board.squeeze(0)
    return board

def reset_game(game):
    """Reset the game state"""
    board = game.get_initial_state()
    # Convert to numpy if needed
    if hasattr(board, 'cpu'):
        board = board.cpu().numpy()
    if board.ndim == 3:
        board = board.squeeze(0)
    st.session_state.board = board
    st.session_state.current_player = 1
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.last_ai_probs = None

def get_cell_display(value, game_name):
    """Get display character for cell value"""
    if game_name == "TicTacToe":
        if value == 1: return "X"
        elif value == -1: return "O"
    else: # Gomoku
        # We use CSS classes for Gomoku stones, so text is empty or minimal
        return "" 
    return ""

def check_game_over(game, board, last_action):
    """Check if game is over and return status"""
    if last_action is not None:
        # Convert board to tensor if needed for GPU game
        if hasattr(game, 'device'):  # GomokuGPU
            if not isinstance(board, torch.Tensor):
                board_tensor = torch.tensor(board, device=game.device, dtype=torch.float32)
                if board_tensor.ndim == 2:
                    board_tensor = board_tensor.unsqueeze(0)  # (H,W) -> (1,H,W)
            else:
                board_tensor = board
            action_tensor = torch.tensor([last_action], device=game.device)
            value, is_terminal = game.get_value_and_terminated(board_tensor.unsqueeze(0), action_tensor)
            value = value.item()
            is_terminal = is_terminal.item()
        else:  # CPU game
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
    
    # Convert board for GPU game
    if hasattr(game, 'device'):  # GomokuGPU
        board_tensor = torch.tensor(st.session_state.board, device=game.device, dtype=torch.float32)
        if board_tensor.ndim == 2:
            board_tensor = board_tensor.unsqueeze(0)  # (H,W) -> (1,H,W)
        ai_state = game.change_perspective(board_tensor.unsqueeze(0), player=-1).squeeze(0)
        ai_state_np = ai_state.cpu().numpy()
    else:
        ai_state_np = game.change_perspective(st.session_state.board.copy(), player=-1)
    
    with st.spinner("AI is thinking..."):
        action_probs = mcts.search(ai_state_np)
    
    st.session_state.last_ai_probs = action_probs.reshape(game.row_count, game.column_count)
    
    # Get valid moves
    if hasattr(game, 'device'):  # GomokuGPU
        board_for_valid = torch.tensor(st.session_state.board, device=game.device, dtype=torch.float32)
        if board_for_valid.ndim == 2:
            board_for_valid = board_for_valid.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)
        valid_moves = game.get_valid_moves(board_for_valid).cpu().numpy().flatten()
    else:
        valid_moves = game.get_valid_moves(st.session_state.board)
        
    action_probs *= valid_moves
    action = np.argmax(action_probs)
    
    # Make move
    if hasattr(game, 'device'):  # GomokuGPU  
        board_tensor = torch.tensor(st.session_state.board, device=game.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        action_tensor = torch.tensor([action], device=game.device)
        player_tensor = torch.tensor([st.session_state.current_player], device=game.device, dtype=torch.float32)
        new_board = game.get_next_state(board_tensor, action_tensor, player_tensor)
        st.session_state.board = new_board.squeeze().cpu().numpy()
    else:
        st.session_state.board = game.get_next_state(
            st.session_state.board, 
            action, 
            st.session_state.current_player
        )
    
    is_over, winner = check_game_over(game, st.session_state.board, action)
    if is_over:
        st.session_state.game_over = True
        st.session_state.winner = winner
    else:
        st.session_state.current_player = game.get_opponent(st.session_state.current_player)

def make_move(row, col, game, mcts):
    """Handle human move"""
    if st.session_state.game_over:
        return
    
    if st.session_state.board[row, col] != 0:
        st.warning("⚠️ Cell already occupied!")
        return
    
    action = row * game.column_count + col
    
    # Make move
    if hasattr(game, 'device'):  # GomokuGPU
        board_tensor = torch.tensor(st.session_state.board, device=game.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        action_tensor = torch.tensor([action], device=game.device)
        player_tensor = torch.tensor([st.session_state.current_player], device=game.device, dtype=torch.float32)
        new_board = game.get_next_state(board_tensor, action_tensor, player_tensor)
        st.session_state.board = new_board.squeeze().cpu().numpy()
    else:
        st.session_state.board = game.get_next_state(
            st.session_state.board,
            action,
            st.session_state.current_player
        )
    
    is_over, winner = check_game_over(game, st.session_state.board, action)
    if is_over:
        st.session_state.game_over = True
        st.session_state.winner = winner
        return
    
    st.session_state.current_player = game.get_opponent(st.session_state.current_player)
    ai_move(game, mcts)

def render_board(board, game_name, key_prefix="game", on_click=None, disabled=False):
    """Render the game board"""
    rows, cols_count = board.shape
    
    # Inject specific styles for this render
    if game_name == "Gomoku":
        # Force square aspect ratio and centering for Gomoku board
        st.markdown("""
        <style>
            .stButton button {
                border-radius: 50%;
            }
        </style>
        """, unsafe_allow_html=True)

    for row in range(rows):
        cols = st.columns(cols_count)
        for col in range(cols_count):
            cell_value = board[row, col]
            
            # Determine button help/label/type
            if game_name == "TicTacToe":
                display = get_cell_display(cell_value, game_name)
                if cell_value == 1: label = f":blue[{display}]"
                elif cell_value == -1: label = f":red[{display}]"
                else: label = " "
                help_text = None
            else:
                # Gomoku: Use empty label but rely on CSS injection via key or state? 
                # Streamlit buttons are hard to style individually without custom components.
                # We will use emoji/unicode for stones if CSS classes aren't easily applied per button.
                # Actually, let's use the 'help' or just unicode circles.
                if cell_value == 1: label = "⚫" 
                elif cell_value == -1: label = "⚪"
                else: label = " "
                help_text = f"({row}, {col})"

            with cols[col]:
                st.button(
                    label,
                    key=f"{key_prefix}_{row}_{col}",
                    disabled=disabled or cell_value != 0,
                    use_container_width=True,
                    on_click=on_click,
                    args=(row, col) if on_click else None,
                    help=help_text
                )

def play_game_ui(game, mcts, device, game_name):
    st.title(f"🎮 AlphaZero {game_name}")
    
    initialize_session_state(game)
    
    # Sidebar info
    st.sidebar.markdown(f"""
    **How to Play:**
    - You are **{'X (Blue)' if game_name == 'TicTacToe' else 'Black (●)'}**
    - AI is **{'O (Red)' if game_name == 'TicTacToe' else 'White (○)'}**
    - Click on empty cells to make your move
    - AI will respond automatically
    """)
    st.sidebar.markdown(f"**Device:** {device}")
    
    # Game status
    if st.session_state.game_over:
        if st.session_state.winner == 1:
            status_class = "status-win"
            status_text = "🎉 You Win!"
        elif st.session_state.winner == -1:
            status_class = "status-win"
            status_text = "🤖 AI Wins!"
        else:
            status_class = "status-draw"
            status_text = "🤝 It's a Draw!"
    else:
        status_class = "status-ongoing"
        if game_name == "TicTacToe":
            current_symbol = "X" if st.session_state.current_player == 1 else "O"
        else:
            current_symbol = "●" if st.session_state.current_player == 1 else "○"
            
        current_name = "Your" if st.session_state.current_player == 1 else "AI's"
        status_text = f"🎯 {current_name} Turn ({current_symbol})"
    
    st.markdown(f'<div class="game-status {status_class}">{status_text}</div>', unsafe_allow_html=True)
    
    # Game board
    # Custom render loop with styling injection
    if game_name == "Gomoku":
        # Tighter spacing for Gomoku
        st.markdown("""
        <style>
            div[data-testid="column"] {
                width: 24px !important;
                flex: 0 0 24px !important;
                min-width: 24px !important;
                padding: 1px !important;
            }
            div[data-testid="stHorizontalBlock"] {
                justify-content: center;
                gap: 1px !important;
            }
            .stButton > button {
                height: 24px !important;
                width: 24px !important;
                padding: 0px !important;
                border-radius: 0px !important;
                border: 1px solid #666;
                background-color: #daa520; /* Gold/Wood color */
                color: black;
                min-height: 24px !important;
                font-size: 16px !important;
            }
            .stButton > button:hover {
                border-color: #333;
                background-color: #c9950f;
            }
            .stButton > button:disabled {
                background-color: #daa520;
                opacity: 1.0;
            }
            /* Black stones - dark background, white border */
            .stButton > button:disabled:has(p:first-child:contains("●")) {
                background-color: #000000 !important;
                border: 2px solid #ffffff !important;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.5);
            }
            /* White stones - white background, dark border  */
            .stButton > button:disabled:has(p:first-child:contains("○")) {
                background-color: #ffffff !important;
                border: 2px solid #000000 !important;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.3);
            }
            .stButton > button p {
                font-size: 26px !important;
                margin: 0px !important;
                line-height: 24px !important;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stButton > button {
                height: 80px !important;
                font-size: 32px !important;
            }
        </style>
        """, unsafe_allow_html=True)

    # Center the board
    with st.container():
        for row in range(game.row_count):
            cols = st.columns(game.column_count)
            for col in range(game.column_count):
                cell_value = st.session_state.board[row, col]
                
                if game_name == "TicTacToe":
                    display = get_cell_display(cell_value, game_name)
                    if cell_value == 1: label = f":blue[{display}]"
                    elif cell_value == -1: label = f":red[{display}]"
                    else: label = " "
                else:
                    # Gomoku - use better symbols
                    if cell_value == 1: 
                        label = "●"  # Filled circle (Black/You)
                    elif cell_value == -1: 
                        label = "○"  # Hollow circle (White/AI)
                    else: 
                        label = " "
                
                with cols[col]:
                    if st.button(
                        label,
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
            reset_game(game)
            st.rerun()
    
    # MCTS Visualization (Only for TicTacToe for now)
    if st.session_state.last_ai_probs is not None and game_name == "TicTacToe":
        st.markdown("---")
        st.markdown("### 🧠 AI's Last Move Analysis")
        # ... (same as before)

def evolution_ui(game_name):
    # ... (Keep existing evolution UI logic)
    st.title(f"📈 Training Evolution - {game_name}")
    
    # Show metrics plot
    plot_path = Path(f"docs/training_plots/{game_name.lower()}_metrics.png")
    if not plot_path.exists():
         plot_path = Path("docs/training_plots/training_metrics.png") # Fallback

    if plot_path.exists():
        st.image(str(plot_path), caption="Training Metrics", use_container_width=True)
        st.markdown("---")
    
    json_path = Path(f"docs/{game_name.lower()}_evolution_replay.json")
    if not json_path.exists():
        # Fallback to default
        json_path = Path("docs/evolution_replay.json")
        
    if not json_path.exists():
        st.error(f"Evolution data not found at {json_path}")
        st.info("Please run `python scripts/generate_evolution_replay.py` first.")
        return

    with open(json_path, "r") as f:
        evolution_data = json.load(f)

    # Sidebar selection
    iterations = [d['iteration'] for d in evolution_data]
    if not iterations:
        st.warning("No evolution data found.")
        return
        
    selected_iter = st.sidebar.select_slider("Select Iteration", options=iterations, value=iterations[-1])
    
    data = next(d for d in evolution_data if d['iteration'] == selected_iter)
    
    st.markdown(f"### Iteration {selected_iter}")
    st.markdown(f"**Model:** `{data['checkpoint']}`")
    
    res_map = {1: "AI Won", -1: "AI Lost", 0: "Draw"}
    st.markdown(f"**Result:** {res_map.get(data['result'], 'Unknown')}")
    
    moves = data['moves']
    
    if 'replay_step' not in st.session_state:
        st.session_state.replay_step = 0
        
    # Ensure step is valid
    if st.session_state.replay_step > len(moves):
        st.session_state.replay_step = 0
        
    # Controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Prev"):
            st.session_state.replay_step = max(0, st.session_state.replay_step - 1)
    with col3:
        if st.button("Next ➡️"):
            st.session_state.replay_step = min(len(moves), st.session_state.replay_step + 1)
            
    step = st.session_state.replay_step
    st.progress(step / len(moves) if len(moves) > 0 else 0)
    st.caption(f"Step {step} / {len(moves)}")
    
    # Get board state
    if step < len(moves):
        current_move = moves[step]
        board = np.array(current_move['board'])
        next_player = current_move['player']
        action = current_move['action']
        
        if action is not None:
            player_name = "AI" if next_player == 1 else "Opponent"
            st.markdown(f"<div class='replay-move'>{player_name} to move...</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='replay-move'>Game Over</div>", unsafe_allow_html=True)
    else:
        # Final state
        last_move = moves[-1]
        if last_move['action'] is None:
            board = np.array(last_move['board'])
            st.markdown("<div class='replay-move'>Final Board</div>", unsafe_allow_html=True)
        else:
            board = np.array(last_move['board'])

    # Render board (read-only)
    render_board(board, game_name, key_prefix="replay", disabled=True)

def main():
    st.sidebar.title("Configuration")
    game_name = st.sidebar.selectbox("Select Game", ["TicTacToe", "Gomoku 9x9", "Gomoku Fixed", "Gomoku Long", "Gomoku 30min", "Gomoku 9x9 GPU Test", "Gomoku 15x15"])
    
    # Checkpoint selector - get available checkpoints
    checkpoint_dir = Path(__file__).parent.parent.parent.parent.parent / "checkpoints" / game_name.lower().replace(" ", "_")
    if not checkpoint_dir.exists():
        checkpoint_dir = Path(__file__).parent.parent.parent.parent.parent / "checkpoints"
    
    checkpoints = sorted(checkpoint_dir.glob("model_*.pt"))
    selected_checkpoint = None
    
    if checkpoints:
        checkpoint_options = ["Latest (Auto)"] + [cp.name for cp in checkpoints]
        selected_checkpoint = st.sidebar.selectbox(
            "Model Checkpoint",
            checkpoint_options,
            help="Select a specific checkpoint or use latest automatically"
        )
    
    # Clear session state if game changes
    if 'current_game' not in st.session_state:
        st.session_state.current_game = game_name
    
    if st.session_state.current_game != game_name:
        st.session_state.current_game = game_name
        st.session_state.board = None # Force reset
        # Clear other state keys
        keys_to_clear = ['board', 'current_player', 'game_over', 'winner', 'last_ai_probs']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    game, model, mcts, device = load_model(game_name, selected_checkpoint)
    
    page = st.sidebar.radio("Navigation", ["Play Game", "Training Evolution"])
    
    if page == "Play Game":
        play_game_ui(game, mcts, device, game_name)
    else:
        evolution_ui(game_name)

if __name__ == "__main__":
    main()
