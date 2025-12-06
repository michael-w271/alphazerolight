
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>

namespace py = pybind11;

// Forward declaration
class Node;

// Game logic interface (simplified for Gomoku)
// We will implement the necessary game logic directly in C++ for speed,
// or callback to Python if it's too complex.
// For Gomoku, checking wins is fast in C++.
// But to keep it simple and compatible with the Python Game class,
// we might want to callback or re-implement.
// Re-implementing check_win in C++ is best for speed.

class GomokuGame {
public:
  int board_size;
  int win_length;

  GomokuGame(int size = 9, int length = 5)
      : board_size(size), win_length(length) {}

  // Check win for a specific player
  // Board is assumed to be a flat vector or pointer
  bool check_win(const float *board, int player) {
    // board is (board_size * board_size)
    // 1 for player, -1 for opponent, 0 for empty
    // But wait, the Python GomokuGPU uses 1 and -1.

    // Horizontal
    for (int r = 0; r < board_size; ++r) {
      for (int c = 0; c <= board_size - win_length; ++c) {
        bool win = true;
        for (int k = 0; k < win_length; ++k) {
          if (board[r * board_size + c + k] != player) {
            win = false;
            break;
          }
        }
        if (win)
          return true;
      }
    }

    // Vertical
    for (int r = 0; r <= board_size - win_length; ++r) {
      for (int c = 0; c < board_size; ++c) {
        bool win = true;
        for (int k = 0; k < win_length; ++k) {
          if (board[(r + k) * board_size + c] != player) {
            win = false;
            break;
          }
        }
        if (win)
          return true;
      }
    }

    // Diagonal (Down-Right)
    for (int r = 0; r <= board_size - win_length; ++r) {
      for (int c = 0; c <= board_size - win_length; ++c) {
        bool win = true;
        for (int k = 0; k < win_length; ++k) {
          if (board[(r + k) * board_size + (c + k)] != player) {
            win = false;
            break;
          }
        }
        if (win)
          return true;
      }
    }

    // Diagonal (Up-Right)
    for (int r = win_length - 1; r < board_size; ++r) {
      for (int c = 0; c <= board_size - win_length; ++c) {
        bool win = true;
        for (int k = 0; k < win_length; ++k) {
          if (board[(r - k) * board_size + (c + k)] != player) {
            win = false;
            break;
          }
        }
        if (win)
          return true;
      }
    }

    return false;
  }

  std::vector<int> get_valid_moves(const float *board) {
    std::vector<int> moves;
    for (int i = 0; i < board_size * board_size; ++i) {
      if (board[i] == 0) {
        moves.push_back(i);
      }
    }
    return moves;
  }
};

class Node {
public:
  std::vector<float> state; // Flat board state
  Node *parent;
  int action_taken;
  float prior;
  int visit_count;
  float value_sum;
  std::vector<std::unique_ptr<Node>> children;
  int player_to_move; // 1 or -1

  Node(const std::vector<float> &s, Node *p, int action, float prior_prob,
       int player)
      : state(s), parent(p), action_taken(action), prior(prior_prob),
        visit_count(0), value_sum(0.0f), player_to_move(player) {}

  bool is_fully_expanded() const { return !children.empty(); }

  float get_ucb(const Node *child, float c_puct) const {
    if (child->visit_count == 0) {
      return 10000.0f + child->prior; // High value for unvisited
    }
    // Use actual Q-value (average reward), not inverted
    float q_value = child->value_sum / child->visit_count;
    float u = c_puct * child->prior * std::sqrt((float)visit_count) /
              (1.0f + child->visit_count);
    return q_value + u;
  }

  Node *select(float c_puct) {
    Node *best_child = nullptr;
    float best_ucb = -std::numeric_limits<float>::infinity();

    for (const auto &child : children) {
      float ucb = get_ucb(child.get(), c_puct);
      if (ucb > best_ucb) {
        best_ucb = ucb;
        best_child = child.get();
      }
    }
    return best_child;
  }

  void expand(const std::vector<float> &policy,
              const std::vector<int> &valid_moves, int board_size) {
    // policy is full size (81), but we only create children for valid moves
    // We also need to compute the next state for each child

    for (int action : valid_moves) {
      float prob = policy[action];
      if (prob > 0) {
        // Create next state
        std::vector<float> next_state = state;
        next_state[action] = (float)player_to_move;

        // Flip perspective for the child node?
        // AlphaZero usually flips perspective so the neural net always sees
        // "me" as 1. If we do that, we need to flip the board values. Let's
        // stick to the logic: Current node state is from perspective of
        // player_to_move. Child node state should be from perspective of
        // -player_to_move.

        // 1. Apply move
        // 2. Flip signs
        for (auto &val : next_state) {
          val = -val;
        }
        // The move we just made (player_to_move) becomes -1 in the new state
        // (opponent) Wait, if we set it to player_to_move (1) then flip, it
        // becomes -1. Correct.

        children.push_back(std::make_unique<Node>(next_state, this, action,
                                                  prob, -player_to_move));
      }
    }
  }

  void backpropagate(float value) {
    value_sum += value;
    visit_count++;

    if (parent != nullptr) {
      // Value is from perspective of player who just moved (parent's
      // player_to_move) So for parent, it is the value. But wait, standard MCTS
      // backprop flips value. If child value is V (for child's player), then
      // for parent it is -V.
      parent->backpropagate(-value);
    }
  }
};

class MCTS_CPP {
  GomokuGame game;
  float c_puct;
  int num_searches;
  int board_size;

public:
  MCTS_CPP(int size, int searches, float c)
      : game(size), c_puct(c), num_searches(searches), board_size(size) {}

  // Batched search
  // states: list of numpy arrays (flat)
  // model_func: python function that takes batch of states and returns
  // (policies, values)
  std::vector<std::vector<float>>
  search_batch(const std::vector<py::array_t<float>> &initial_states,
               py::object model_func) {
    int batch_size = initial_states.size();
    std::vector<std::unique_ptr<Node>> roots;
    roots.reserve(batch_size);

    // Initialize roots
    for (const auto &state_np : initial_states) {
      auto r = state_np.unchecked<1>();
      std::vector<float> state_vec(r.size());
      for (int i = 0; i < r.size(); ++i)
        state_vec[i] = r(i);

      // Root is always player 1 to move in canonical form
      roots.push_back(std::make_unique<Node>(state_vec, nullptr, -1, 0.0f, 1));
    }

    for (int search = 0; search < num_searches; ++search) {
      // 1. Selection
      std::vector<Node *> leaf_nodes;
      std::vector<int> leaf_indices;

      // Release GIL for parallel selection
      {
        py::gil_scoped_release release;

// Parallel Selection
#pragma omp parallel
        {
          std::vector<Node *> local_leaves;
          std::vector<int> local_indices;

#pragma omp for nowait
          for (int i = 0; i < batch_size; ++i) {
            Node *node = roots[i].get();
            while (node->is_fully_expanded()) {
              node = node->select(c_puct);
            }

            bool terminated = false;
            float value = 0.0f;

            if (game.check_win(node->state.data(), -1)) {
              terminated = true;
              value = -1.0f;
            } else if (game.get_valid_moves(node->state.data()).empty()) {
              terminated = true;
              value = 0.0f;
            }

            if (terminated) {
              node->backpropagate(value);
            } else {
              local_leaves.push_back(node);
              local_indices.push_back(i);
            }
          }

#pragma omp critical
          {
            leaf_nodes.insert(leaf_nodes.end(), local_leaves.begin(),
                              local_leaves.end());
            leaf_indices.insert(leaf_indices.end(), local_indices.begin(),
                                local_indices.end());
          }
        }
      } // GIL re-acquired here

      if (leaf_nodes.empty())
        continue;

      // 2. Prepare batch for NN
      int num_leaves = leaf_nodes.size();
      int area = board_size * board_size;

      std::vector<float> batch_data(num_leaves * 3 * area);

      {
        py::gil_scoped_release release;
#pragma omp parallel for
        for (int i = 0; i < num_leaves; ++i) {
          const auto &s = leaf_nodes[i]->state;
          for (int j = 0; j < area; ++j) {
            float val = s[j];
            batch_data[i * 3 * area + 0 * area + j] =
                (val == -1.0f) ? 1.0f : 0.0f;
            batch_data[i * 3 * area + 1 * area + j] =
                (val == 0.0f) ? 1.0f : 0.0f;
            batch_data[i * 3 * area + 2 * area + j] =
                (val == 1.0f) ? 1.0f : 0.0f;
          }
        }
      }

      // Call Python model (Needs GIL)
      py::array_t<float> input_np({num_leaves, 3, board_size, board_size},
                                  batch_data.data());
      py::tuple result = model_func(input_np);

      py::array_t<float> policies_np = result[0].cast<py::array_t<float>>();
      py::array_t<float> values_np = result[1].cast<py::array_t<float>>();

      auto policies_r = policies_np.unchecked<2>();
      auto values_r = values_np.unchecked<2>();

      // 3. Expansion and Backprop
      {
        py::gil_scoped_release release;
#pragma omp parallel for
        for (int i = 0; i < num_leaves; ++i) {
          Node *node = leaf_nodes[i];
          float value = values_r(i, 0);

          std::vector<int> valid_moves =
              game.get_valid_moves(node->state.data());

          std::vector<float> policy(area);
          float policy_sum = 0.0f;

          for (int move : valid_moves) {
            policy[move] = policies_r(i, move);
            policy_sum += policy[move];
          }

          if (policy_sum > 0) {
            for (int move : valid_moves) {
              policy[move] /= policy_sum;
            }
          } else {
            float uniform = 1.0f / valid_moves.size();
            for (int move : valid_moves) {
              policy[move] = uniform;
            }
          }

          node->expand(policy, valid_moves, board_size);
          node->backpropagate(value);
        }
      }
    }

    // Return action probabilities
    std::vector<std::vector<float>> results;
    results.reserve(batch_size);

    for (const auto &root : roots) {
      std::vector<float> probs(board_size * board_size, 0.0f);
      float sum = 0.0f;
      for (const auto &child : root->children) {
        probs[child->action_taken] = (float)child->visit_count;
        sum += child->visit_count;
      }
      if (sum > 0) {
        for (auto &p : probs)
          p /= sum;
      }
      results.push_back(probs);
    }

    return results;
  }
};

PYBIND11_MODULE(mcts_cpp, m) {
  py::class_<MCTS_CPP>(m, "MCTS_CPP")
      .def(py::init<int, int, float>())
      .def("search_batch", &MCTS_CPP::search_batch);
}
