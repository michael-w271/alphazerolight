#include "telemetry_client.h"
#include <zmq.h>
#include <iostream>
#include <chrono>
#include <thread>

namespace azl {

TelemetryClient::TelemetryClient(const std::string& endpoint)
    : endpoint_(endpoint),
      zmq_context_(nullptr),
      zmq_socket_(nullptr),
      running_(false),
      connected_(false),
      frames_received_(0),
      metrics_received_(0),
      net_summaries_received_(0) {
}

TelemetryClient::~TelemetryClient() {
    stop();
}

void TelemetryClient::start() {
    if (running_.load()) {
        return;  // Already running
    }

    // Initialize ZeroMQ context and socket
    zmq_context_ = zmq_ctx_new();
    if (!zmq_context_) {
        std::cerr << "Failed to create ZMQ context" << std::endl;
        return;
    }

    zmq_socket_ = zmq_socket(zmq_context_, ZMQ_SUB);
    if (!zmq_socket_) {
        std::cerr << "Failed to create ZMQ socket" << std::endl;
        zmq_ctx_destroy(zmq_context_);
        return;
    }

    // Subscribe to all messages (empty filter)
    zmq_setsockopt(zmq_socket_, ZMQ_SUBSCRIBE, "", 0);

    // Set receive timeout to 1000ms (non-blocking with timeout)
    int timeout = 1000;
    zmq_setsockopt(zmq_socket_, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));

    // Connect to endpoint
    if (zmq_connect(zmq_socket_, endpoint_.c_str()) != 0) {
        std::cerr << "Failed to connect to " << endpoint_ << std::endl;
        zmq_close(zmq_socket_);
        zmq_ctx_destroy(zmq_context_);
        return;
    }

    std::cout << "📡 Telemetry client connected to " << endpoint_ << std::endl;

    running_ = true;
    receive_thread_ = std::make_unique<std::thread>(&TelemetryClient::receive_loop, this);
}

void TelemetryClient::stop() {
    if (!running_.load()) {
        return;
    }

    running_ = false;

    if (receive_thread_ && receive_thread_->joinable()) {
        receive_thread_->join();
    }

    if (zmq_socket_) {
        zmq_close(zmq_socket_);
        zmq_socket_ = nullptr;
    }

    if (zmq_context_) {
        zmq_ctx_destroy(zmq_context_);
        zmq_context_ = nullptr;
    }

    std::cout << "📡 Telemetry client stopped" << std::endl;
}

void TelemetryClient::receive_loop() {
    char buffer[65536];  // 64KB buffer for JSON messages

    while (running_.load()) {
        int size = zmq_recv(zmq_socket_, buffer, sizeof(buffer) - 1, 0);

        if (size > 0) {
            buffer[size] = '\0';  // Null-terminate
            connected_ = true;

            try {
                parse_and_store(std::string(buffer, size));
            } catch (const std::exception& e) {
                std::cerr << "Error parsing telemetry message: " << e.what() << std::endl;
            }
        } else if (size == -1 && zmq_errno() == EAGAIN) {
            // Timeout - no message received
            connected_ = false;
        } else if (size == -1) {
            // Error
            std::cerr << "ZMQ receive error: " << zmq_strerror(zmq_errno()) << std::endl;
            connected_ = false;
        }
    }
}

void TelemetryClient::parse_and_store(const std::string& json_str) {
    auto j = json::parse(json_str);
    std::string type = j["type"];

    if (type == "frame") {
        FrameMessage frame;
        frame.timestamp_ms = j["timestamp_ms"];
        frame.iteration = j["iteration"];
        frame.game_idx = j["game_idx"];
        frame.move_idx = j["move_idx"];
        frame.player = j["player"];
        frame.board = j["board"].get<std::vector<std::vector<int>>>();
        frame.valid_moves = j["valid_moves"].get<std::vector<int>>();
        frame.policy_head = j["policy_head"].get<std::vector<float>>();
        frame.value_head = j["value_head"];
        frame.mcts_policy = j["mcts_policy"].get<std::vector<float>>();
        frame.root_visits = j["root_visits"].get<std::vector<int>>();
        frame.root_q = j["root_q"].get<std::vector<float>>();
        frame.chosen_action = j["chosen_action"];
        frame.temperature = j["temperature"];
        frame.is_terminal = j["is_terminal"];
        frame.terminal_value = j.value("terminal_value", 0.0f);

        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_frame_ = frame;
            frame_history_.push_back(frame);
            if (frame_history_.size() > MAX_FRAME_HISTORY) {
                frame_history_.pop_front();
            }
        }

        frames_received_++;

    } else if (type == "metrics") {
        MetricsMessage metrics;
        metrics.timestamp_ms = j["timestamp_ms"];
        metrics.iteration = j["iteration"];
        metrics.total_loss = j["total_loss"];
        metrics.policy_loss = j["policy_loss"];
        metrics.value_loss = j["value_loss"];
        metrics.policy_entropy = j.value("policy_entropy", 0.0f);
        metrics.examples_seen = j.value("examples_seen", 0);
        metrics.eval_winrate = j.value("eval_winrate", 0.0f);
        metrics.avg_game_length = j.value("avg_game_length", 0.0f);

        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            latest_metrics_ = metrics;
        }

        metrics_received_++;

    } else if (type == "net_summary") {
        NetSummaryMessage summary;
        summary.timestamp_ms = j["timestamp_ms"];
        summary.iteration = j["iteration"];

        if (j.contains("layer_norms") && !j["layer_norms"].is_null()) {
            summary.layer_norms = j["layer_norms"].get<std::map<std::string, float>>();
        }

        if (j.contains("activation_norms") && !j["activation_norms"].is_null()) {
            summary.activation_norms = j["activation_norms"].get<std::map<std::string, float>>();
        }

        {
            std::lock_guard<std::mutex> lock(net_summary_mutex_);
            latest_net_summary_ = summary;
        }

        net_summaries_received_++;
    }
}

bool TelemetryClient::get_latest_frame(FrameMessage& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (frames_received_.load() == 0) {
        return false;
    }
    frame = latest_frame_;
    return true;
}

bool TelemetryClient::get_latest_metrics(MetricsMessage& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    if (metrics_received_.load() == 0) {
        return false;
    }
    metrics = latest_metrics_;
    return true;
}

bool TelemetryClient::get_latest_net_summary(NetSummaryMessage& summary) {
    std::lock_guard<std::mutex> lock(net_summary_mutex_);
    if (net_summaries_received_.load() == 0) {
        return false;
    }
    summary = latest_net_summary_;
    return true;
}

std::vector<FrameMessage> TelemetryClient::get_frame_history(size_t count) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    if (frame_history_.empty()) {
        return {};
    }

    size_t start_idx = 0;
    if (frame_history_.size() > count) {
        start_idx = frame_history_.size() - count;
    }

    return std::vector<FrameMessage>(
        frame_history_.begin() + start_idx,
        frame_history_.end()
    );
}

}  // namespace azl
