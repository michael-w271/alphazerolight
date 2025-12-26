#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <deque>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace azl {

// Frame message: per-move game state + MCTS thinking
struct FrameMessage {
    int64_t timestamp_ms;
    int iteration;
    int game_idx;
    int move_idx;
    int player;
    std::vector<std::vector<int>> board;  // 6x7 for Connect Four
    std::vector<int> valid_moves;
    std::vector<float> policy_head;
    float value_head;
    std::vector<float> mcts_policy;
    std::vector<int> root_visits;
    std::vector<float> root_q;
    int chosen_action;
    float temperature;
    bool is_terminal;
    float terminal_value;
    float policy_entropy;       // NEW: Shannon entropy of policy
    float mcts_improvement;     // NEW: How much MCTS improved over raw policy
};

// Metrics message: training stats per iteration
struct MetricsMessage {
    int64_t timestamp_ms;
    int iteration;
    float total_loss;
    float policy_loss;
    float value_loss;
    float policy_entropy;
    int examples_seen;
    float eval_winrate;
    float avg_game_length;
    std::map<std::string, float> gradient_norms;  // NEW: Gradient norms by layer
    float policy_sharpness;                        // NEW: Policy confidence metric
    float mcts_improvement_avg;                    // NEW: Average MCTS improvement
};

// Network summary message: layer norms and activations  
struct NetSummaryMessage {
    int64_t timestamp_ms;
    int iteration;
    std::map<std::string, float> layer_norms;
    std::map<std::string, float> activation_norms;
};

/**
 * Telemetry client - receives ZeroMQ messages from Python trainer
 * Runs background thread for receiving, pushes to ring buffer
 */
class TelemetryClient {
public:
    explicit TelemetryClient(const std::string& endpoint);
    ~TelemetryClient();

    // Start/stop background receiver thread
    void start();
    void stop();

    // Get latest messages (thread-safe)
    bool get_latest_frame(FrameMessage& frame);
    bool get_latest_metrics(MetricsMessage& metrics);
    bool get_latest_net_summary(NetSummaryMessage& summary);

    // Get frame history (for replay)
    std::vector<FrameMessage> get_frame_history(size_t count = 100);

    // Statistics
    int get_frames_received() const { return frames_received_.load(); }
    int get_metrics_received() const { return metrics_received_.load(); }
    bool is_connected() const { return connected_.load(); }

private:
    void receive_loop();
    void parse_and_store(const std::string& json_str);

    std::string endpoint_;
    void* zmq_context_;
    void* zmq_socket_;

    std::atomic<bool> running_;
    std::atomic<bool> connected_;
    std::unique_ptr<std::thread> receive_thread_;

    // Thread-safe storage
    std::mutex frame_mutex_;
    std::mutex metrics_mutex_;
    std::mutex net_summary_mutex_;

    FrameMessage latest_frame_;
    MetricsMessage latest_metrics_;
    NetSummaryMessage latest_net_summary_;

    std::deque<FrameMessage> frame_history_;  // Ring buffer
    static constexpr size_t MAX_FRAME_HISTORY = 1000;

    std::atomic<int> frames_received_;
    std::atomic<int> metrics_received_;
    std::atomic<int> net_summaries_received_;
};

}  // namespace azl
